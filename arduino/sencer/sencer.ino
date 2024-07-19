#define outPin A0
#define sensorPin  3

void setup() {
    Serial.begin(115200);
    pinMode(outPin, INPUT);
    pinMode(sensorPin, INPUT);
}

void loop() {
  int sensorValue = digitalRead(sensorPin);  
  Serial.print(sensorValue); 
  Serial.print(","); 

  int analogValue = analogRead(outPin);
  
  // アナログ値を電圧に変換
  float voltage = analogValue * (5.0 / 1023.0);
  
  // 電圧を距離に変換（1mレンジの場合）
  float distance = (voltage / 5.0) * 100; // 5Vが100cmに相当

  // シリアルモニタに出力
  Serial.println(distance);
   
  delay(100); // 100msごとに測定
}
