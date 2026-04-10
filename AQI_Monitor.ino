// ─────────────────────────────────────────────────────────────────────────────
// Air Quality Monitor
// Author : George Reigner T  (20C-J027444)
// Hardware: Arduino Uno, PMS5003, DHT22, DS3231 RTC, SD Card, 16x2 I2C LCD
// Logs PM2.5, PM10, Temperature, Humidity, AQI to SD card as CSV
// CSV is directly compatible with the LSTM Python training script
// ─────────────────────────────────────────────────────────────────────────────

#include <SoftwareSerial.h>
#include <DHT.h>
#include <SD.h>
#include <RTClib.h>
#include <LiquidCrystal_I2C.h>

// ── Pin Definitions ───────────────────────────────────────────────────────────
#define PMS_RX   2          // PMS5003 TX → Arduino pin 2
#define PMS_TX   3          // PMS5003 RX → Arduino pin 3
#define DHTPIN   4          // DHT data pin
#define DHTTYPE  DHT22      // Change to DHT11 if using DHT11
#define SD_CS    10         // SD card chip select

// ── Timing ────────────────────────────────────────────────────────────────────
#define READ_INTERVAL   5000   // Sensor read interval (ms)
#define LCD_SLIDE_TIME  3000   // Time each LCD screen is shown (ms)

// ── CSV filename ──────────────────────────────────────────────────────────────
#define LOG_FILE "aqi_log.csv"

// ── Hardware Objects ──────────────────────────────────────────────────────────
SoftwareSerial    pmsSerial(PMS_RX, PMS_TX);
DHT               dht(DHTPIN, DHTTYPE);
RTC_DS3231        rtc;
LiquidCrystal_I2C lcd(0x27, 16, 2);

// ── State Tracking ────────────────────────────────────────────────────────────
unsigned long lastReadTime  = 0;
unsigned long lastSlideTime = 0;
uint8_t       lcdScreen     = 0;

// Last valid sensor readings — shared between read and display
int   pm2_5       = 0;
int   pm10        = 0;
float temperature = 0.0;
float humidity    = 0.0;
int   aqi         = 0;
bool  dataReady   = false;

// ── Function Prototypes ───────────────────────────────────────────────────────
bool    readPMSData(int &pm1_0, int &pm2_5, int &pm10);
int     calculateAQI(int pm2_5, int pm10);
void    updateLCD();
void    logToSD(DateTime now);
void    initSDWithHeader();
void    showStartupScreen();
bool    checkSensors();

// =============================================================================
// SETUP
// =============================================================================
void setup() {
  Serial.begin(9600);
  pmsSerial.begin(9600);
  dht.begin();
  lcd.init();
  lcd.backlight();

  showStartupScreen();

  // RTC check
  if (!rtc.begin()) {
    Serial.println(F("ERROR: RTC not found"));
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print(F("RTC Error"));
    lcd.setCursor(0, 1); lcd.print(F("Check wiring"));
    while (1);  // Halt — no point continuing without time
  }

  // Optional: uncomment once to set RTC time, then comment out again
  // rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));

  // SD card check
  if (!SD.begin(SD_CS)) {
    Serial.println(F("ERROR: SD card failed"));
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print(F("SD Card Error"));
    lcd.setCursor(0, 1); lcd.print(F("Check card"));
    while (1);  // Halt — logging is core to this project
  }

  Serial.println(F("SD card OK"));
  initSDWithHeader();

  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(F("System Ready"));
  lcd.setCursor(0, 1); lcd.print(F("Logging..."));
  delay(2000);
  lcd.clear();

  Serial.println(F("Setup complete. Monitoring started."));
}

// =============================================================================
// MAIN LOOP — non-blocking using millis()
// =============================================================================
void loop() {
  unsigned long now = millis();

  // ── Sensor read every READ_INTERVAL ms ─────────────────────────────────────
  if (now - lastReadTime >= READ_INTERVAL) {
    lastReadTime = now;

    int pm1_0_raw, pm2_5_raw, pm10_raw;

    if (readPMSData(pm1_0_raw, pm2_5_raw, pm10_raw)) {
      float tempRead = dht.readTemperature();
      float humRead  = dht.readHumidity();

      if (isnan(tempRead) || isnan(humRead)) {
        Serial.println(F("WARNING: DHT read failed, skipping this cycle"));
      } else {
        // Update shared state
        pm2_5       = pm2_5_raw;
        pm10        = pm10_raw;
        temperature = tempRead;
        humidity    = humRead;
        aqi         = calculateAQI(pm2_5, pm10);
        dataReady   = true;

        DateTime timestamp = rtc.now();

        // Log to serial monitor
        Serial.print(F("Time: "));     Serial.print(timestamp.timestamp(DateTime::TIMESTAMP_FULL));
        Serial.print(F(" | PM2.5: ")); Serial.print(pm2_5);
        Serial.print(F(" | PM10: "));  Serial.print(pm10);
        Serial.print(F(" | Temp: "));  Serial.print(temperature, 1);
        Serial.print(F(" | Hum: "));   Serial.print(humidity, 1);
        Serial.print(F(" | AQI: "));   Serial.println(aqi);

        // Log to SD card
        logToSD(timestamp);
      }
    } else {
      Serial.println(F("WARNING: PMS5003 read failed, skipping this cycle"));
    }
  }

  // ── LCD slide every LCD_SLIDE_TIME ms (non-blocking) ───────────────────────
  if (dataReady && (now - lastSlideTime >= LCD_SLIDE_TIME)) {
    lastSlideTime = now;
    updateLCD();
  }
}

// =============================================================================
// FUNCTIONS
// =============================================================================

// ── Startup splash screen ─────────────────────────────────────────────────────
void showStartupScreen() {
  lcd.setCursor(0, 0); lcd.print(F("Air Quality"));
  lcd.setCursor(0, 1); lcd.print(F("Monitor v1.0"));
  delay(2000);
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(F("George Reigner"));
  lcd.setCursor(0, 1); lcd.print(F("20C-J027444"));
  delay(3000);
  lcd.clear();
}

// ── Create CSV with header if file does not already exist ─────────────────────
void initSDWithHeader() {
  if (!SD.exists(LOG_FILE)) {
    File f = SD.open(LOG_FILE, FILE_WRITE);
    if (f) {
      f.println(F("Timestamp,PM2.5,PM10,Temperature,Humidity,AQI"));
      f.close();
      Serial.println(F("CSV header written"));
    } else {
      Serial.println(F("ERROR: Could not create log file"));
    }
  } else {
    Serial.println(F("Existing log file found, appending"));
  }
}

// ── Log one row to SD card CSV ────────────────────────────────────────────────
void logToSD(DateTime now) {
  File f = SD.open(LOG_FILE, FILE_WRITE);
  if (f) {
    f.print(now.timestamp(DateTime::TIMESTAMP_FULL)); f.print(",");
    f.print(pm2_5);                                   f.print(",");
    f.print(pm10);                                    f.print(",");
    f.print(temperature, 1);                          f.print(",");
    f.print(humidity, 1);                             f.print(",");
    f.println(aqi);
    f.close();
  } else {
    Serial.println(F("ERROR: Failed to open log file for writing"));
  }
}

// ── Cycle through LCD screens without blocking ────────────────────────────────
void updateLCD() {
  lcd.clear();

  switch (lcdScreen) {
    case 0:
      lcd.setCursor(0, 0); lcd.print(F("PM2.5:"));
      lcd.setCursor(7, 0); lcd.print(pm2_5); lcd.print(F(" ug/m3"));
      lcd.setCursor(0, 1); lcd.print(F("PM10: "));
      lcd.setCursor(7, 1); lcd.print(pm10);  lcd.print(F(" ug/m3"));
      break;

    case 1:
      lcd.setCursor(0, 0); lcd.print(F("Temp: "));
      lcd.setCursor(6, 0); lcd.print(temperature, 1); lcd.print(F(" C"));
      lcd.setCursor(0, 1); lcd.print(F("Hum:  "));
      lcd.setCursor(6, 1); lcd.print(humidity, 1);    lcd.print(F(" %"));
      break;

    case 2:
      lcd.setCursor(0, 0); lcd.print(F("AQI: ")); lcd.print(aqi);
      lcd.setCursor(0, 1);
      if      (aqi <= 50)  lcd.print(F("Good"));
      else if (aqi <= 100) lcd.print(F("Moderate"));
      else if (aqi <= 150) lcd.print(F("Unhealthy-SG"));
      else if (aqi <= 200) lcd.print(F("Unhealthy"));
      else                 lcd.print(F("Hazardous"));
      break;
  }

  lcdScreen = (lcdScreen + 1) % 3;  // Cycle 0 → 1 → 2 → 0
}

// ── Read 32-byte frame from PMS5003 ──────────────────────────────────────────
bool readPMSData(int &pm1_0, int &pm2_5, int &pm10) {
  if (pmsSerial.available() < 32) return false;

  uint8_t buffer[32];
  pmsSerial.readBytes(buffer, 32);

  // Validate start bytes
  if (buffer[0] != 0x42 || buffer[1] != 0x4D) return false;

  // Validate checksum
  uint16_t checksum = 0;
  for (int i = 0; i < 30; i++) checksum += buffer[i];
  uint16_t received = (buffer[30] << 8) | buffer[31];
  if (checksum != received) {
    Serial.println(F("WARNING: PMS checksum mismatch"));
    return false;
  }

  // Atmospheric environment values (bytes 10-15)
  pm1_0 = (buffer[10] << 8) | buffer[11];
  pm2_5 = (buffer[12] << 8) | buffer[13];
  pm10  = (buffer[14] << 8) | buffer[15];

  return true;
}

// ── AQI calculation based on US EPA breakpoints ───────────────────────────────
// Uses linear interpolation between breakpoints for accuracy
int calculateAQI(int pm2_5, int pm10) {

  // PM2.5 breakpoints: {Clow, Chigh, Ilow, Ihigh}
  float pm25_bp[][4] = {
    {0.0,  12.0,  0,  50},
    {12.1, 35.4,  51, 100},
    {35.5, 55.4,  101, 150},
    {55.5, 150.4, 151, 200},
    {150.5, 250.4, 201, 300},
    {250.5, 500.4, 301, 500}
  };

  // PM10 breakpoints
  float pm10_bp[][4] = {
    {0,   54,  0,   50},
    {55,  154, 51,  100},
    {155, 254, 101, 150},
    {255, 354, 151, 200},
    {355, 424, 201, 300},
    {425, 604, 301, 500}
  };

  int aqi_pm25 = 0, aqi_pm10 = 0;

  for (int i = 0; i < 6; i++) {
    if (pm2_5 <= pm25_bp[i][1]) {
      aqi_pm25 = (int)((pm25_bp[i][3] - pm25_bp[i][2]) /
                 (pm25_bp[i][1] - pm25_bp[i][0]) *
                 ((float)pm2_5 - pm25_bp[i][0]) + pm25_bp[i][2]);
      break;
    }
  }

  for (int i = 0; i < 6; i++) {
    if (pm10 <= pm10_bp[i][1]) {
      aqi_pm10 = (int)((pm10_bp[i][3] - pm10_bp[i][2]) /
                 (pm10_bp[i][1] - pm10_bp[i][0]) *
                 ((float)pm10 - pm10_bp[i][0]) + pm10_bp[i][2]);
      break;
    }
  }

  return max(aqi_pm25, aqi_pm10);
}
