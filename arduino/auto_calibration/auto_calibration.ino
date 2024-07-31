#include <Servo.h>

Servo myservo;

void setup() {
  myservo.attach(9);
  Serial.begin(9600);
}

void loop() {
  
  if (Serial.available() > 0) {
    int angle = Serial.parseInt();  // Pythonから送られてきたステップ数を取得

    if(angle >=0 && angle < 180){
      myStepper.step(steps);
      myservo.write(angle);
    }
  }

  delay(100); 

}
