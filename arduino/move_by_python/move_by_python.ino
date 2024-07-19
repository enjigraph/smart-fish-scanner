#include <Stepper.h>

const int dirPin = 11;  // 方向ピン
const int stepPin = 9;  // ステップピン
const int stepsPerRevolution = 800;  // ステッピングモーターの1回転あたりのステップ数

Stepper myStepper(stepsPerRevolution,stepPin,dirPin);

void setup() {
  // シリアル通信の開始
  Serial.begin(115200);
  myStepper.setSpeed(800);

  // ピンのモード設定
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);

}

void loop() {
  if (Serial.available() > 0) {
    int steps = Serial.parseInt();  // Pythonから送られてきたステップ数を取得

    if(steps != 0){
      myStepper.step(steps);
      Serial.print("motion completed:");
      Serial.println(steps);
    }
  }

  delay(100); 

}
