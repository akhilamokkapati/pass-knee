// Geared motor + hall-effect encoder test (BTS7960B, XIAO ESP32-S3).
// Same hold-to-run button pattern as twist_untwist.cpp, no current sensing.
// Encoder is quadrature (C1/C2) - direction from which channel leads.
// M1/M2 go to the BTS7960B motor output, not to the ESP32.

#include <Arduino.h>

const int RPWM = D0;
const int LPWM = D1;
const int R_EN = D2;
const int L_EN = D3;

const int button1 = D8; // LEFT / reverse
const int button2 = D9; // RIGHT / forward

const int ENCODER_C1 = D10;
const int ENCODER_C2 = D4;

const int MAX_DUTY = 80; // unvalidated starting point
const int RAMP_STEP_MS = 15;

volatile long encoderPosition = 0;

void IRAM_ATTR onEncoderC1Rise() {
  if (digitalRead(ENCODER_C2) == HIGH) {
    encoderPosition++;
  } else {
    encoderPosition--;
  }
}

unsigned long lastPrintMs = 0;
long lastEncoderPosition = 0;
const unsigned long PRINT_INTERVAL_MS = 200;

// called from loop() and from inside runHeld()'s wait, so it doesn't
// freeze while a button is held
void printEncoderStatsIfDue() {
  unsigned long now = millis();
  if (now - lastPrintMs < PRINT_INTERVAL_MS) return;

  long position = encoderPosition;
  long delta = position - lastEncoderPosition;
  lastEncoderPosition = position;
  float countsPerSec = delta * 1000.0 / PRINT_INTERVAL_MS;

  Serial.print("time ");
  Serial.print(now);
  Serial.print(',');
  Serial.print("   pos ");
  Serial.print(position);
  Serial.print(',');
  Serial.print("   counts "); 
  Serial.println(countsPerSec);

  lastPrintMs = now;
}

void fullStop() {
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);
}

void softRamp(int pwmPin, int targetDuty) {
  for (int duty = 0; duty <= targetDuty; duty += 5) {
    analogWrite(pwmPin, duty);
    delay(RAMP_STEP_MS);
  }
}

void softRampDown(int pwmPin, int startDuty) {
  for (int duty = startDuty; duty >= 0; duty -= 5) {
    analogWrite(pwmPin, duty);
    delay(RAMP_STEP_MS);
  }
}

void runHeld(int pwmPin, int buttonPin, const __FlashStringHelper* label) {
  Serial.print(label);
  Serial.println(F(" pressed, holding..."));

  softRamp(pwmPin, MAX_DUTY);

  while (digitalRead(buttonPin) == HIGH) {
    printEncoderStatsIfDue();
    delay(1);
  }

  softRampDown(pwmPin, MAX_DUTY);
  fullStop();

  Serial.print(label);
  Serial.println(F(" released."));
}

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  pinMode(R_EN, OUTPUT);
  pinMode(L_EN, OUTPUT);

  pinMode(button1, INPUT);
  pinMode(button2, INPUT);

  pinMode(ENCODER_C1, INPUT);
  pinMode(ENCODER_C2, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENCODER_C1), onEncoderC1Rise, RISING);

  digitalWrite(R_EN, HIGH);
  digitalWrite(L_EN, HIGH);

  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);

  Serial.println(F("t_ms,encoder_position,counts_per_s"));
}

void loop() {
  bool leftPressed  = (digitalRead(button1) == HIGH);
  bool rightPressed = (digitalRead(button2) == HIGH);

  if (leftPressed) {
    runHeld(LPWM, button1, F("LEFT"));
  } else if (rightPressed) {
    runHeld(RPWM, button2, F("RIGHT"));
  }

  printEncoderStatsIfDue();
}
