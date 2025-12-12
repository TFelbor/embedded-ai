extern "C" {
#include "model.h"
}

#define MAX_MSG_SIZE 32768

void setup() {

  // Initialize serial port
  Serial.begin(115200);

  // Increase the timeout
  Serial.setTimeout(10000);

  // Initialize pin for blinking LED
  pinMode(PIN_LED, OUTPUT);

  // Wait for initialization
  while (!Serial && millis() < 5000);

  // Notify readyness
  Serial.println("READY");
}

void loop() {
  static unsigned int inference_count = 0;
  static char buf[MAX_MSG_SIZE];
  
  // Use a single flat buffer for inputs to match Keras Flatten()
  static number_t inputs[MODEL_INPUT_TOTAL]; 
  static number_t outputs[MODEL_OUTPUT_SAMPLES];

  // 1. Read Data
  int msg_len = Serial.readBytesUntil('\n', buf, MAX_MSG_SIZE);
  if (msg_len < 1) { Serial.println("READY"); return; }
  if (msg_len != MAX_MSG_SIZE) msg_len++;

  // 2. Parse and Convert
  char *pbuf = buf;
  for (int i = 0; i < MODEL_INPUT_TOTAL; i++) {
    // Parse float
    float val_float = strtof(pbuf, &pbuf);
    pbuf++; // Skip comma
    
    // Scale to Fixed Point (Q9)
    // No loop re-ordering needed anymore!
    long_number_t val_fixed = (long_number_t)(val_float * (1 << FIXED_POINT));
    inputs[i] = clamp_to_number_t(val_fixed);
  }

  // 3. Run Inference
  digitalWrite(PIN_LED, HIGH);
  cnn(inputs, outputs);
  digitalWrite(PIN_LED, LOW);

  // 4. Get Prediction (Argmax)
  unsigned int label = 0;
  float max_val = outputs[0];
  for (unsigned int i = 1; i < MODEL_OUTPUT_SAMPLES; i++) {
    if (max_val < outputs[i]) {
      max_val = outputs[i];
      label = i;
    }
  }

  // 5. Send Result
  inference_count++;
  char msg[64];
  snprintf(msg, sizeof(msg), "%d,%d,%f", inference_count, label, (double)max_val);
  Serial.println(msg);
}
