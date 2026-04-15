#include <SPI.h>
#include <WiFiNINA.h>
#include <WiFiUdp.h>
#include <string.h>
#include <stdlib.h>

const char* WIFI_SSID = "Glide_Resident";
const char* WIFI_PASS = "ScootJulyClash";

const unsigned int LOCAL_CMD_PORT = 6006;
WiFiUDP udp;

char packetBuffer[128];

// LED output pins 
const int SIGNAL_PIN_0 = 2; // signal0 -> D2
const int SIGNAL_PIN_1 = 3; // signal1 -> D3
const int SIGNAL_PIN_2 = 4; // signal2 -> D4
const int SIGNAL_PIN_3 = 5; // signal3 -> D5
const int SIGNAL_PIN_4 = 6; // signal4 -> D6
const int SIGNAL_PIN_5 = 7; // signal5 -> D7

// Pulse duration: 1.5 seconds
const unsigned long SIGNAL_HIGH_MS = 1500;

// Connect to Wi-Fi
void connectToWiFi() {
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(WIFI_SSID);

  while (WiFi.status() != WL_CONNECTED) {
    WiFi.begin(WIFI_SSID, WIFI_PASS);

    unsigned long startAttempt = millis();
    while (millis() - startAttempt < 10000) {
      if (WiFi.status() == WL_CONNECTED) {
        break;
      }
      delay(500);
      Serial.print(".");
    }
    Serial.println();

    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("Wi-Fi connection failed. Retrying...");
      delay(2000);
    }
  }

  Serial.println("Wi-Fi connected.");
  Serial.print("Local IP: ");
  Serial.println(WiFi.localIP());
}

// Initialize LED output pins
void initSignalPins() {
  pinMode(SIGNAL_PIN_0, OUTPUT);
  pinMode(SIGNAL_PIN_1, OUTPUT);
  pinMode(SIGNAL_PIN_2, OUTPUT);
  pinMode(SIGNAL_PIN_3, OUTPUT);
  pinMode(SIGNAL_PIN_4, OUTPUT);
  pinMode(SIGNAL_PIN_5, OUTPUT);

  digitalWrite(SIGNAL_PIN_0, LOW);
  digitalWrite(SIGNAL_PIN_1, LOW);
  digitalWrite(SIGNAL_PIN_2, LOW);
  digitalWrite(SIGNAL_PIN_3, LOW);
  digitalWrite(SIGNAL_PIN_4, LOW);
  digitalWrite(SIGNAL_PIN_5, LOW);
}

// Output a 1.5s HIGH pulse on selected pin
void pulsePin(int pin, const char* name) {
  Serial.print("Trigger ");
  Serial.print(name);
  Serial.print(" on pin D");
  Serial.println(pin);

  digitalWrite(pin, HIGH);
  delay(SIGNAL_HIGH_MS);
  digitalWrite(pin, LOW);

  Serial.print(name);
  Serial.println(" pulse finished.");
}

// Execute signal action
// Each trigger outputs HIGH for 1.5s, then LOW
void handleSignal(int signalNum) {
  Serial.print("handleSignal() called with signal ");
  Serial.println(signalNum);

  switch (signalNum) {
    case 0:
      pulsePin(SIGNAL_PIN_0, "signal0");
      break;

    case 1:
      pulsePin(SIGNAL_PIN_1, "signal1");
      break;

    case 2:
      pulsePin(SIGNAL_PIN_2, "signal2");
      break;

    case 3:
      pulsePin(SIGNAL_PIN_3, "signal3");
      break;

    case 4:
      pulsePin(SIGNAL_PIN_4, "signal4");
      break;

    case 5:
      pulsePin(SIGNAL_PIN_5, "signal5");
      break;

    default:
      Serial.println("Unknown signal number");
      break;
  }
}

// Parse received command
// Supported formats:
//   "0" ~ "5"
//   "signal0" ~ "signal5"
//   numeric string 0~5
bool parseAndHandleCommand(const char* cmd) {
  if (cmd == nullptr) {
    return false;
  }

  int len = strlen(cmd);
  if (len <= 0) {
    return false;
  }

  // Case 1: single character "0" ~ "5"
  if (len == 1 && cmd[0] >= '0' && cmd[0] <= '5') {
    int signalNum = cmd[0] - '0';
    Serial.print("Parsed single-digit signal: ");
    Serial.println(signalNum);
    handleSignal(signalNum);
    return true;
  }

  // Case 2: "signal0" ~ "signal5"
  if (strncmp(cmd, "signal", 6) == 0 && len == 7) {
    char c = cmd[6];
    if (c >= '0' && c <= '5') {
      int signalNum = c - '0';
      Serial.print("Parsed named signal: ");
      Serial.println(signalNum);
      handleSignal(signalNum);
      return true;
    }
  }

  // Case 3: numeric string
  char* endPtr = nullptr;
  long value = strtol(cmd, &endPtr, 10);
  if (endPtr != cmd && *endPtr == '\0') {
    if (value >= 0 && value <= 5) {
      Serial.print("Parsed numeric string signal: ");
      Serial.println((int)value);
      handleSignal((int)value);
      return true;
    }
  }

  Serial.print("Invalid command: ");
  Serial.println(cmd);
  return false;
}

// Remove trailing newline / carriage return / spaces
void trimTrailingWhitespace(char* s) {
  if (s == nullptr) return;

  int len = strlen(s);
  while (len > 0) {
    char c = s[len - 1];
    if (c == '\r' || c == '\n' || c == ' ' || c == '\t') {
      s[len - 1] = '\0';
      len--;
    } else {
      break;
    }
  }
}

// Setup
void setup() {
  Serial.begin(115200);
  delay(1500);

  Serial.println("=== MKR1010 Wi-Fi UDP Receiver Start ===");

  initSignalPins();
  connectToWiFi();

  if (!udp.begin(LOCAL_CMD_PORT)) {
    Serial.println("UDP begin failed!");
    while (1) {}
  }

  Serial.print("Listening for UDP commands on port ");
  Serial.println(LOCAL_CMD_PORT);
  Serial.println("Waiting for command from Python...");
}

// Loop
void loop() {
  // Auto reconnect Wi-Fi if disconnected
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi disconnected. Reconnecting...");
    connectToWiFi();

    udp.stop();
    if (!udp.begin(LOCAL_CMD_PORT)) {
      Serial.println("UDP re-begin failed!");
      delay(1000);
      return;
    }

    Serial.print("UDP listening resumed on port ");
    Serial.println(LOCAL_CMD_PORT);
  }

  int packetSize = udp.parsePacket();
  if (packetSize > 0) {
    if (packetSize >= (int)sizeof(packetBuffer)) {
      packetSize = sizeof(packetBuffer) - 1;
    }

    int len = udp.read(packetBuffer, packetSize);
    if (len > 0) {
      packetBuffer[len] = '\0';
    } else {
      packetBuffer[0] = '\0';
    }

    trimTrailingWhitespace(packetBuffer);

    Serial.print("Received UDP packet from ");
    Serial.print(udp.remoteIP());
    Serial.print(":");
    Serial.println(udp.remotePort());

    Serial.print("Received raw command: ");
    Serial.println(packetBuffer);

    bool ok = parseAndHandleCommand(packetBuffer);
    if (ok) {
      Serial.println("Command handled successfully.");
    } else {
      Serial.println("Command handling failed.");
    }

    Serial.println("------------------------------");
  }
}
