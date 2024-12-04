#include <HTTPClient.h>
#include <WiFi.h>

#include "driver/i2s.h" // Include I2S driver

// Wi-Fi credentials
const char* ssid = "Sudarshan M34";
const char* password = "Warriors";

// Server URL
const char* serverURL = "http://192.168.216.221:5000/upload";

unsigned long prev_time;

// I2S pins
#define I2S_SCK 14 // Bit clock (BCLK)
#define I2S_SD 32 // Serial data (DOUT)
#define I2S_WS 15 // Word select (LRCL)

// Configuration for 1-second audio chunk
#define SAMPLE_RATE 16000 // Set sample rate to 16kHz for the INMP441
#define CHUNK_DURATION 1 // 1 second duration
#define CHUNK_SIZE (SAMPLE_RATE * CHUNK_DURATION) // Number of samples for 1 second

#define SEC2MILLIS 1000

int16_t audioBuffer[CHUNK_SIZE]; // Buffer to store 1 second of audio

void setup() {
    Serial.begin(115200);
    WiFi.begin(ssid, password);

    // Wait for Wi-Fi connection
    while (WiFi.status() != WL_CONNECTED) {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");

    // I2S configuration for INMP441
    i2s_config_t i2s_config = {
        .mode = i2s_mode_t(I2S_MODE_MASTER | I2S_MODE_RX), // Master mode, receive
        .sample_rate = SAMPLE_RATE, // 16kHz sample rate
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT, // 16-bit per sample
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT, // Mono input (only left channel)
        .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S), // Standard I2S format
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, // Interrupt level 1
        .dma_buf_count = 8, // Number of DMA buffers
        .dma_buf_len = 1024, // Length of each DMA buffer
        .use_apll = false, // Not using APLL
    };

    // I2S pin configuration
    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK, // BCLK pin
        .ws_io_num = I2S_WS, // LRCL pin
        .data_out_num = I2S_PIN_NO_CHANGE, // No data output
        .data_in_num = I2S_SD // DOUT pin (serial data input)
    };

    // Install and start the I2S driver
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);

    // Set the I2S clock
    i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_16BIT, I2S_CHANNEL_MONO);
    
    prev_time = millis();
}

void loop() {
    unsigned long curr_time = millis();

    if ((curr_time - prev_time) > (CHUNK_DURATION * SEC2MILLIS)) {
      prev_time = curr_time;

      size_t bytes_read;

      // Capture 1 second of audio data from the INMP441 microphone
      i2s_read(I2S_NUM_0, (void*)audioBuffer, sizeof(audioBuffer), &bytes_read, portMAX_DELAY);

      Serial.printf("Read %d bytes of audio.\n", bytes_read);
      
      // Send the 1-second audio data via HTTP
      HTTPClient http;
      http.begin(serverURL);
      http.addHeader("Content-Type", "application/octet-stream");

      // Convert buffer to bytes and send
      uint8_t* byteData = (uint8_t*)audioBuffer;
      int httpResponseCode = http.POST(byteData, bytes_read);


      if (httpResponseCode > 0) {
          Serial.printf("Data sent, response code: %d\n", httpResponseCode);
      }
      
      else {
          Serial.printf("Error in sending data: %s\n", http.errorToString(httpResponseCode).c_str());
      }

      http.end(); // End the connection
    }

    // delay(1000); // Wait 1 second before capturing and sending again
}
