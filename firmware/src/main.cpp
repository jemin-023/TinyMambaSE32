#include <arduinoFFT.h>

#define N 512  // samples per frame

int16_t samples[N];
float re[N], im[N];
ArduinoFFT<float> fft(re, im, N, 16000.0f);

void setup() {
  Serial.setRxBufferSize(2048);  // default is too small for one frame
  Serial.begin(115200);
}

// laptop sends N int16 samples, we reply with N/2 magnitudes + FFT time in us
void loop() {
  if (Serial.readBytes((char *)samples, sizeof(samples)) != sizeof(samples)) return;

  for (int i = 0; i < N; i++) {
    re[i] = samples[i] / 32768.0f;
    im[i] = 0;
  }
  uint32_t t0 = micros();
  fft.compute(FFTDirection::Forward);
  fft.complexToMagnitude();  // magnitudes end up in re[]
  uint32_t us = micros() - t0;

  Serial.write((uint8_t *)re, N / 2 * sizeof(float));
  Serial.write((uint8_t *)&us, sizeof(us));
}
