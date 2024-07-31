#include <Servo.h>

#define BLIN1 0
#define BLIN2 1
#define FLIN1 2
#define FLIN2 3
#define FRIN1 8
#define FRIN2 9
#define BRIN1 10
#define BRIN2 11


Servo servo1;
Servo servo2;


void setup() {
  pinMode(BLIN1,OUTPUT);
  pinMode(BLIN2,OUTPUT);
  pinMode(FLIN1,OUTPUT);
  pinMode(FLIN2,OUTPUT);
  pinMode(FRIN1,OUTPUT);
  pinMode(FRIN2,OUTPUT);
  pinMode(BRIN1,OUTPUT);
  pinMode(BRIN2,OUTPUT);


  servo1.attach(7);
  servo2.attach(6);


  Serial.begin(9600);
}


void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');  // 改行までの文字列を読み取る


    int commaIndex1 = command.indexOf(',');
    int commaIndex2 = command.indexOf(',', commaIndex1 + 1);


    if (commaIndex1 == -1 || commaIndex2 == -1) {
      return; // コマンドが不正
    }


    int servo1Angle = command.substring(0, commaIndex1).toInt();
    int servo2Angle = command.substring(commaIndex1 + 1, commaIndex2).toInt();
    String direction = command.substring(commaIndex2 + 1);


    controlServo(1, servo1Angle);
    controlServo(2, servo2Angle);
    controlMotors(direction);


  }
}


void controlServo(int servo, int angle) {
  if (servo == 1) {
    servo1.write(angle);
  } else if (servo == 2) {
    servo2.write(angle);
  }
}


void controlMotors(String direction) {


  if (direction == "forward") {


    digitalWrite(BLIN1,LOW);
    digitalWrite(BLIN2,HIGH);
    digitalWrite(FLIN1,LOW);
    digitalWrite(FLIN2,HIGH);
    digitalWrite(FRIN1,LOW);
    digitalWrite(FRIN2,HIGH);
    digitalWrite(BRIN1,LOW);
    digitalWrite(BRIN2,HIGH);
    delay(1000);


  } else if (direction == "backward") {


    digitalWrite(BLIN1,HIGH);
    digitalWrite(BLIN2,LOW);
    digitalWrite(FLIN1,HIGH);
    digitalWrite(FLIN2,LOW);
    digitalWrite(FRIN1,HIGH);
    digitalWrite(FRIN2,LOW);
    digitalWrite(BRIN1,HIGH);
    digitalWrite(BRIN2,LOW);
    delay(1000);


  } else if (direction == "left") {


    digitalWrite(BLIN1,LOW);
    digitalWrite(BLIN2,HIGH);
    digitalWrite(FLIN1,HIGH);
    digitalWrite(FLIN2,LOW);
    digitalWrite(FRIN1,LOW);
    digitalWrite(FRIN2,HIGH);
    digitalWrite(BRIN1,HIGH);
    digitalWrite(BRIN2,LOW);
    delay(1000);


  } else if (direction == "right") {


    digitalWrite(BLIN1,HIGH);
    digitalWrite(BLIN2,LOW);
    digitalWrite(FLIN1,LOW);
    digitalWrite(FLIN2,HIGH);
    digitalWrite(FRIN1,HIGH);
    digitalWrite(FRIN2,LOW);
    digitalWrite(BRIN1,LOW);
    digitalWrite(BRIN2,HIGH);
    delay(1000);


  }


  digitalWrite(BLIN1,LOW);
  digitalWrite(BLIN2,LOW);
  digitalWrite(FLIN1,LOW);
  digitalWrite(FLIN2,LOW);
  digitalWrite(FRIN1,LOW);
  digitalWrite(FRIN2,LOW);
  digitalWrite(BRIN1,LOW);
  digitalWrite(BRIN2,LOW);


}
