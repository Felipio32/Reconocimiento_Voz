#include <Arduino.h>
#include <driver/i2s.h>

// Configuración de pines para el micrófono INMP441
#define I2S_WS 22
#define I2S_SD 21
#define I2S_SCK 26

// Configuración de pines para LEDs
#define LED_ADELANTE 32
#define LED_ATRAS 27
#define LED_DERECHA 33
#define LED_IZQUIERDA 25

// Configuración de audio
#define SAMPLE_RATE 16000
#define BUFFER_SIZE 1024
int16_t sampleBuffer[BUFFER_SIZE];

// Parámetros de detección - AJUSTAR SEGÚN ENTORNO
#define NUM_BANDS 4          // Número de bandas de frecuencia a analizar
#define ENERGY_THRESHOLD 2000000  // Umbral para detectar voz vs silencio

// Estructura para comandos de voz
typedef struct {
  const char* name;
  int ledPin;
  float lowFreqEnergy;    // Energía relativa en frecuencias bajas (0-1kHz)
  float midFreqEnergy;    // Energía relativa en frecuencias medias (1-2kHz)
  float highFreqEnergy;   // Energía relativa en frecuencias altas (2-4kHz)
  float veryHighFreqEnergy; // Energía relativa en frecuencias muy altas (4-8kHz)
} VoiceCommand;

// Definición de comandos con sus características de frecuencia
// Estos valores son aproximados y deben calibrarse
VoiceCommand commands[4] = {
  {"adelante",  LED_ADELANTE, 0.3, 0.5, 1.0, 0.2},  // Frecuencias altas dominantes
  {"atras",     LED_ATRAS,    0.5, 1.0, 0.4, 0.1},  // Frecuencias medias dominantes
  {"derecha",   LED_DERECHA,  0.4, 0.7, 0.7, 0.3},  // Mezcla equilibrada
  {"izquierda", LED_IZQUIERDA,1.0, 0.6, 0.3, 0.1}   // Frecuencias bajas dominantes
};

// Variables de control
unsigned long lastDetectionTime = 0;
unsigned long ledOffTime = 0;
int activeLed = -1;
bool calibrationMode = false;

// Función para inicializar I2S
void initI2S() {
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(4, 2, 0)
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_STAND_I2S),
#else
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S),
#endif
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = BUFFER_SIZE / 8,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };
  
  i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = -1,
    .data_in_num = I2S_SD
  };

  esp_err_t result = i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  if (result != ESP_OK) {
    Serial.printf("Error al instalar driver I2S: %d\n", result);
    return;
  }
  
  result = i2s_set_pin(I2S_NUM_0, &pin_config);
  if (result != ESP_OK) {
    Serial.printf("Error al configurar pines I2S: %d\n", result);
    return;
  }
  
  Serial.println("I2S inicializado correctamente");
}

// Capturar audio del micrófono
bool captureAudio() {
  size_t bytesRead = 0;
  esp_err_t result = i2s_read(I2S_NUM_0, sampleBuffer, BUFFER_SIZE * sizeof(int16_t), &bytesRead, portMAX_DELAY);
  
  if (result != ESP_OK) {
    Serial.printf("Error al leer I2S: %d\n", result);
    return false;
  }
  
  return true;
}

// Calcular la energía total del audio
float calculateTotalEnergy() {
  float energy = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    energy += abs(sampleBuffer[i]);
  }
  return energy;
}

// Análisis simplificado de frecuencias sin usar FFT
void analyzeFrequencies(float* bandEnergies) {
  // Inicializar bandas de energía
  for (int i = 0; i < NUM_BANDS; i++) {
    bandEnergies[i] = 0;
  }
  
  // Método simplificado: contar cruces por cero para diferentes segmentos
  // Este método es una aproximación muy básica pero suficiente para distinguir algunas palabras
  int zeroCrossings[NUM_BANDS] = {0, 0, 0, 0};
  
  // Dividir el buffer en 4 segmentos y contar cruces por cero
  int segmentSize = BUFFER_SIZE / NUM_BANDS;
  
  for (int band = 0; band < NUM_BANDS; band++) {
    int startIdx = band * segmentSize;
    int endIdx = startIdx + segmentSize;
    
    // Contar cruces por cero en este segmento
    for (int i = startIdx + 1; i < endIdx; i++) {
      if ((sampleBuffer[i] >= 0 && sampleBuffer[i-1] < 0) || 
          (sampleBuffer[i] < 0 && sampleBuffer[i-1] >= 0)) {
        zeroCrossings[band]++;
      }
      
      // Acumular energía para este segmento
      bandEnergies[band] += abs(sampleBuffer[i]);
    }
    
    // Normalizar energía
    bandEnergies[band] /= segmentSize;
  }
  
  // Ajustar bandas según cruces por cero (aproximación de frecuencia)
  // Más cruces por cero = frecuencias más altas
  for (int i = 0; i < NUM_BANDS; i++) {
    float crossingFactor = zeroCrossings[i] / (float)segmentSize;
    bandEnergies[i] *= (1.0 + crossingFactor); // Dar más peso a frecuencias altas
  }
  
  // Normalizar todas las bandas respecto al máximo
  float maxEnergy = 0;
  for (int i = 0; i < NUM_BANDS; i++) {
    if (bandEnergies[i] > maxEnergy) {
      maxEnergy = bandEnergies[i];
    }
  }
  
  if (maxEnergy > 0) {
    for (int i = 0; i < NUM_BANDS; i++) {
      bandEnergies[i] /= maxEnergy;
    }
  }
}

// Reconocer comando basado en el patrón de energía
int recognizeCommand(float* bandEnergies) {
  int bestCommand = -1;
  float bestScore = 0.65; // Umbral mínimo de similitud (ajustar según precisión deseada)
  
  for (int i = 0; i < 4; i++) {
    // Calcular similitud entre patrón detectado y patrón de referencia
    float similarity = 1.0 - (
      abs(bandEnergies[0] - commands[i].lowFreqEnergy) +
      abs(bandEnergies[1] - commands[i].midFreqEnergy) +
      abs(bandEnergies[2] - commands[i].highFreqEnergy) +
      abs(bandEnergies[3] - commands[i].veryHighFreqEnergy)
    ) / 4.0;
    
    if (similarity > bestScore) {
      bestScore = similarity;
      bestCommand = i;
    }
  }
  
  // Mostrar puntuación de similitud para depuración
  if (bestCommand >= 0) {
    Serial.printf("Palabra detectada: %s (similitud: %.2f)\n", 
                  commands[bestCommand].name, bestScore);
  }
  
  return bestCommand;
}

// Calibración manual - usar para ajustar umbrales
void calibrateSystem() {
  Serial.println("\n=== MODO DE CALIBRACIÓN ===");
  Serial.println("Este modo te ayudará a ajustar los perfiles para cada palabra.");
  
  for (int cmdIdx = 0; cmdIdx < 4; cmdIdx++) {
    Serial.printf("\nPrepárese para decir '%s' en 3 segundos...\n", commands[cmdIdx].name);
    for (int i = 3; i > 0; i--) {
      Serial.printf("%d... ", i);
      delay(1000);
    }
    Serial.println("¡AHORA!");
    
    // Capturar y analizar
    if (captureAudio()) {
      float totalEnergy = calculateTotalEnergy();
      
      if (totalEnergy > ENERGY_THRESHOLD) {
        float bandEnergies[NUM_BANDS];
        analyzeFrequencies(bandEnergies);
        
        // Mostrar resultados
        Serial.printf("Patrón detectado para '%s':\n", commands[cmdIdx].name);
        Serial.printf("Bajas: %.2f, Medias: %.2f, Altas: %.2f, Muy altas: %.2f\n",
                     bandEnergies[0], bandEnergies[1], bandEnergies[2], bandEnergies[3]);
        
        Serial.println("Copia estos valores en el código:");
        Serial.printf("{\"%s\", LED_%s, %.2f, %.2f, %.2f, %.2f},\n",
                    commands[cmdIdx].name, commands[cmdIdx].name,
                    bandEnergies[0], bandEnergies[1], bandEnergies[2], bandEnergies[3]);
        
        delay(2000); // Esperar antes de la siguiente palabra
      } else {
        Serial.println("No se detectó suficiente volumen. Intente de nuevo.");
        cmdIdx--; // Repetir esta palabra
      }
    }
  }
  
  Serial.println("\nCalibración completada. Copia los valores en el código para un mejor reconocimiento.");
}

// Encender LED correspondiente
void activateLED(int commandIndex) {
  // Apagar todos los LEDs primero
  digitalWrite(LED_ADELANTE, LOW);
  digitalWrite(LED_ATRAS, LOW);
  digitalWrite(LED_DERECHA, LOW);
  digitalWrite(LED_IZQUIERDA, LOW);
  
  if (commandIndex >= 0 && commandIndex < 4) {
    digitalWrite(commands[commandIndex].ledPin, HIGH);
    Serial.printf("Comando detectado: %s (LED: %d)\n", 
                  commands[commandIndex].name, commands[commandIndex].ledPin);
    
    activeLed = commands[commandIndex].ledPin;
    ledOffTime = millis() + 2000; // Mantener el LED encendido por 2 segundos
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\nSistema de Reconocimiento de Voz ESP32 (Simple)");
  
  // Configurar pines de LEDs
  pinMode(LED_ADELANTE, OUTPUT);
  pinMode(LED_ATRAS, OUTPUT);
  pinMode(LED_DERECHA, OUTPUT);
  pinMode(LED_IZQUIERDA, OUTPUT);
  
  // Secuencia de prueba de LEDs
  digitalWrite(LED_ADELANTE, HIGH);
  delay(200);
  digitalWrite(LED_ADELANTE, LOW);
  digitalWrite(LED_ATRAS, HIGH);
  delay(200);
  digitalWrite(LED_ATRAS, LOW);
  digitalWrite(LED_DERECHA, HIGH);
  delay(200);
  digitalWrite(LED_DERECHA, LOW);
  digitalWrite(LED_IZQUIERDA, HIGH);
  delay(200);
  digitalWrite(LED_IZQUIERDA, LOW);
  
  // Inicializar I2S
  initI2S();
  
  // Verificar si se debe entrar en modo de calibración
  // (conectar GPIO 15 a GND para activar)
  pinMode(15, INPUT_PULLUP);
  if (digitalRead(15) == LOW) {
    calibrationMode = true;
    calibrateSystem();
  }
  
  Serial.println("Sistema listo. Diga 'adelante', 'atras', 'derecha' o 'izquierda'");
}

void loop() {
  // Capturar audio
  if (captureAudio()) {
    // Calcular energía total
    float totalEnergy = calculateTotalEnergy();
    
    // Detectar voz vs silencio
    if (totalEnergy > ENERGY_THRESHOLD) {
      // Solo procesar si ha pasado tiempo suficiente desde la última detección
      if (millis() - lastDetectionTime > 1000) {
        Serial.printf("Energía: %.2f\n", totalEnergy);
        
        // Analizar frecuencias
        float bandEnergies[NUM_BANDS];
        analyzeFrequencies(bandEnergies);
        
        // Mostrar bandas de energía (para depuración)
        Serial.printf("Bandas: %.2f %.2f %.2f %.2f\n", 
                     bandEnergies[0], bandEnergies[1], bandEnergies[2], bandEnergies[3]);
        
        // Reconocer comando
        int commandIndex = recognizeCommand(bandEnergies);
        
        // Activar LED si se detectó un comando
        if (commandIndex >= 0) {
          activateLED(commandIndex);
          lastDetectionTime = millis();
        }
      }
    }
  }
  
  // Apagar LED si ha pasado el tiempo
  if (activeLed >= 0 && millis() > ledOffTime) {
    digitalWrite(activeLed, LOW);
    activeLed = -1;
  }
  
  // Pequeña pausa para estabilidad
  delay(10);
}
