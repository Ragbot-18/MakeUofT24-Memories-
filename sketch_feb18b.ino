#include <Wire.h>
#include <rgb_lcd.h>
#include <Servo.h>

rgb_lcd lcd;
Servo myServo;

// LED pin
const int ledPin = 6;

// Buzzer pin
const int buzzerPin = 5;

String s;

void setup() {
    Serial.begin(9600);
    lcd.begin(16, 2); // Initialize the LCD with 16 characters and 2 lines
    myServo.attach(9); // Attach the servo to pin 9

    pinMode(ledPin, OUTPUT); // Initialize LED pin as output
    pinMode(buzzerPin, OUTPUT); // Initialize buzzer pin as output

    // Initial setup actions like turning the LED off
    digitalWrite(ledPin, LOW);
}

void loop() {
    // Example usage
    while(Serial.available()){
      char x = Serial.read();
      s.concat(x);
      //x = Serial.readString();
      //Serial.print(x);
      //lcd.print(x);
    }
    displayScrollingMessage(s, 200);
    rotateServo(90);
    toggleLED(); // Toggle LED on then off
    playTone(1000, 500); // 1000 Hz for 500 ms
    delay(2); // Wait for 2 seconds before repeating
    s = "";
}


void displayScrollingMessage(String message, int delayMs) {
    message = "                " + message + "                "; // Pad message for smooth scrolling
    int length = message.length();
    for (int position = 0; position < length - 16; ++position) {
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print(message.substring(position, position + 16));
        delay(delayMs);
    }
}

void rotateServo(int angle) {
    myServo.write(angle); // Rotate servo to specified angle
}

void toggleLED() {
    static bool ledState = false;
    ledState = !ledState;
    digitalWrite(ledPin, ledState); // Toggle the LED state
}

void playTone(int tone, int duration) {
    for (long i = 0; i < duration * 1000L; i += tone * 2) {
        digitalWrite(buzzerPin, HIGH);
        delayMicroseconds(tone);
        digitalWrite(buzzerPin, LOW);
        delayMicroseconds(tone);
    }
}
