#include <Arduino.h>
#include <driver/i2s.h>
#include "modelo_comandos_tflite.h"

// TensorFlow Lite Micro
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

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
#define BUFFER_SIZE 16000  // 1 segundo de audio a 16kHz
int16_t sampleBuffer[BUFFER_SIZE];
float audioFeatures[BUFFER_SIZE];

// Variables para TensorFlow Lite Micro
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input_tensor = nullptr;
  TfLiteTensor* output_tensor = nullptr;

  // Arena para tensors de TFLite (ajustar según necesidad)
  constexpr int kTensorArenaSize = 96 * 1024;
  uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
}

// Nombres de los comandos (actualizar según tu modelo)
const int NUM_CLASSES = 4;
const char* commandNames[NUM_CLASSES] = {"adelante", "atras", "derecha", "izquierda"};
const int commandLeds[NUM_CLASSES] = {LED_ADELANTE, LED_ATRAS, LED_DERECHA, LED_IZQUIERDA};

// Variables de control
unsigned long lastDetectionTime = 0;
unsigned long ledOffTime = 0;
int activeLed = -1;
bool isFirstRun = true;

// Umbral de detección para el clasificador
const float DETECTION_THRESHOLD = 0.6f;

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
    .dma_buf_len = 1024,
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

// Inicializar TensorFlow Lite Micro
bool initTFLM() {
  // Configurar reporte de errores
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  
  Serial.println("Inicializando TensorFlow Lite Micro...");
  
  // Cargar modelo
  model = tflite::GetModel(g_model);
  
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Error: versión del modelo (%d) no coincide con la versión de la API (%d).\n",
                  model->version(), TFLITE_SCHEMA_VERSION);
    return false;
  }
  
  Serial.println("Modelo cargado correctamente.");
  
  // Opciones para resolver operaciones
  // Método 1: Todos los operadores (más grande, pero más compatible)
  static tflite::AllOpsResolver resolver;
  
  // Método 2: Solo los operadores necesarios (más eficiente)
  // Descomenta este bloque y comenta el AllOpsResolver si quieres usar solo los ops necesarios
  /*
  static tflite::MicroMutableOpResolver<10> resolver;
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddAveragePool2D();
  resolver.AddMaxPool2D();
  resolver.AddRelu();
  resolver.AddQuantize();
  resolver.AddDequantize();
  */
  
  // Crear intérprete
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter
  );
  interpreter = &static_interpreter;
  
  // Asignar tensores
  Serial.println("Asignando tensores...");
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("Error al asignar tensores.");
    return false;
  }
  
  // Obtener tensores de entrada y salida
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);
  
  // Verificar formato de entrada
  Serial.print("Tensor de entrada: ");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.printf("%d ", input_tensor->dims->data[i]);
  }
  Serial.println();
  
  Serial.print("Tipo de datos de entrada: ");
  switch (input_tensor->type) {
    case kTfLiteFloat32: Serial.println("float32"); break;
    case kTfLiteInt8: Serial.println("int8 (cuantizado)"); break;
    default: Serial.printf("otro tipo (%d)\n", input_tensor->type);
  }
  
  Serial.print("Tensor de salida: ");
  for (int i = 0; i < output_tensor->dims->size; i++) {
    Serial.printf("%d ", output_tensor->dims->data[i]);
  }
  Serial.println();
  
  return true;
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

// Calcular la energía total del audio para detección de actividad
float calculateTotalEnergy() {
  float energy = 0;
  for (int i = 0; i < BUFFER_SIZE; i++) {
    energy += abs(sampleBuffer[i]);
  }
  return energy / BUFFER_SIZE;
}

// Preprocesar audio para adaptarlo al formato del modelo
void preprocessAudio() {
  // Normalizar a rango [-1, 1] para entrenamiento típico
  for (int i = 0; i < BUFFER_SIZE; i++) {
    audioFeatures[i] = sampleBuffer[i] / 32768.0f;
    // Asegurar que se mantiene en el rango [-1, 1]
    if (audioFeatures[i] > 1.0f) audioFeatures[i] = 1.0f;
    if (audioFeatures[i] < -1.0f) audioFeatures[i] = -1.0f;
  }
}

// Copiar audio preprocesado al tensor de entrada
bool copyToInputTensor() {
  try {
    // Verificar tipo de tensor de entrada
    if (input_tensor->type == kTfLiteFloat32) {
      // Para modelo float32 (no cuantizado)
      float* input_data = input_tensor->data.f;
      
      // Determinar cuántos elementos se pueden copiar
      int input_size = 1;
      for (int i = 0; i < input_tensor->dims->size; i++) {
        input_size *= input_tensor->dims->data[i];
      }
      
      // Copiar teniendo en cuenta el tamaño del tensor
      int copy_size = min(input_size, BUFFER_SIZE);
      for (int i = 0; i < copy_size; i++) {
        input_data[i] = audioFeatures[i];
      }
      
      // Rellenar con ceros si es necesario
      for (int i = copy_size; i < input_size; i++) {
        input_data[i] = 0.0f;
      }
      
    } else if (input_tensor->type == kTfLiteInt8) {
      // Para modelo int8 (cuantizado)
      int8_t* input_data = input_tensor->data.int8;
      
      // Determinar cuántos elementos se pueden copiar
      int input_size = 1;
      for (int i = 0; i < input_tensor->dims->size; i++) {
        input_size *= input_tensor->dims->data[i];
      }
      
      // Copiar teniendo en cuenta el tamaño del tensor
      int copy_size = min(input_size, BUFFER_SIZE);
      for (int i = 0; i < copy_size; i++) {
        // Convertir de float [-1, 1] a int8 [-128, 127]
        input_data[i] = (int8_t)(audioFeatures[i] * 127.0f);
      }
      
      // Rellenar con ceros si es necesario
      for (int i = copy_size; i < input_size; i++) {
        input_data[i] = 0;
      }
      
    } else {
      Serial.printf("Tipo de tensor no soportado: %d\n", input_tensor->type);
      return false;
    }
    
    return true;
  } catch (const std::exception& e) {
    Serial.printf("Error al copiar al tensor: %s\n", e.what());
    return false;
  }
}

// Encender LED correspondiente
void activateLED(int commandIndex) {
  // Apagar todos los LEDs primero
  digitalWrite(LED_ADELANTE, LOW);
  digitalWrite(LED_ATRAS, LOW);
  digitalWrite(LED_DERECHA, LOW);
  digitalWrite(LED_IZQUIERDA, LOW);
  
  if (commandIndex >= 0 && commandIndex < NUM_CLASSES) {
    digitalWrite(commandLeds[commandIndex], HIGH);
    Serial.printf("Comando detectado: %s (LED: %d)\n", 
                  commandNames[commandIndex], commandLeds[commandIndex]);
    
    activeLed = commandLeds[commandIndex];
    ledOffTime = millis() + 2000; // Mantener el LED encendido por 2 segundos
  }
}

void setup() {
  // Inicializar comunicación serial
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n\nSistema de Reconocimiento de Voz ESP32 (TensorFlow Lite Micro)");
  
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
  
  // Inicializar I2S para micrófono
  Serial.println("Inicializando I2S...");
  initI2S();
  
  // Inicializar TensorFlow Lite Micro
  Serial.println("Inicializando TensorFlow Lite Micro...");
  if (!initTFLM()) {
    Serial.println("Error al inicializar TensorFlow Lite Micro. El sistema no funcionará correctamente.");
    // No detenemos la ejecución para permitir depuración
  } else {
    Serial.println("TensorFlow Lite Micro inicializado correctamente.");
  }
  
  isFirstRun = true;
  Serial.println("Sistema listo. Diga uno de los comandos: adelante, atras, derecha, izquierda");
}

void loop() {
  // Si es la primera ejecución, hacemos una inferencia de prueba
  if (isFirstRun) {
    Serial.println("Ejecutando inferencia de prueba para verificar el modelo...");
    
    // Llenamos el tensor de entrada con ceros para la prueba
    if (input_tensor->type == kTfLiteFloat32) {
      for (int i = 0; i < input_tensor->bytes / sizeof(float); i++) {
        input_tensor->data.f[i] = 0.0f;
      }
    } else if (input_tensor->type == kTfLiteInt8) {
      for (int i = 0; i < input_tensor->bytes / sizeof(int8_t); i++) {
        input_tensor->data.int8[i] = 0;
      }
    }
    
    // Ejecutamos la inferencia de prueba
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      Serial.println("Error en inferencia de prueba. Verificar modelo.");
    } else {
      Serial.println("Inferencia de prueba exitosa.");
      
      // Mostramos los valores de salida de la inferencia vacía
      Serial.println("Valores de salida en reposo:");
      for (int i = 0; i < NUM_CLASSES; i++) {
        float value = 0.0f;
        if (output_tensor->type == kTfLiteFloat32) {
          value = output_tensor->data.f[i];
        } else if (output_tensor->type == kTfLiteInt8) {
          value = output_tensor->data.int8[i] / 127.0f;
        }
        Serial.printf("  %s: %.6f\n", commandNames[i], value);
      }
    }
    
    isFirstRun = false;
  }
  
  // Capturar audio
  if (captureAudio()) {
    // Calcular energía total
    float totalEnergy = calculateTotalEnergy();
    
    // Umbral de energía para detectar voz (ajustar según entorno)
    const float ENERGY_THRESHOLD = 500.0f;
    
    // Detectar voz vs silencio
    if (totalEnergy > ENERGY_THRESHOLD) {
      // Solo procesar si ha pasado tiempo suficiente desde la última detección
      if (millis() - lastDetectionTime > 1000) {
        Serial.printf("Energía: %.2f\n", totalEnergy);
        
        // Preprocesar audio para el modelo
        preprocessAudio();
        
        // Copiar al tensor de entrada
        if (!copyToInputTensor()) {
          Serial.println("Error al copiar datos al tensor de entrada.");
          delay(100);
          return;
        }
        
        // Realizar inferencia (clasificación)
        unsigned long startTime = millis();
        TfLiteStatus invoke_status = interpreter->Invoke();
        unsigned long inferenceTime = millis() - startTime;
        
        if (invoke_status != kTfLiteOk) {
          Serial.println("Error al ejecutar la inferencia.");
          delay(100);
          return;
        }
        
        Serial.printf("Inferencia completada en %lu ms\n", inferenceTime);
        
        // Procesar resultados
        int bestCommand = -1;
        float bestScore = DETECTION_THRESHOLD;
        
        // Mostrar predicciones y encontrar la mejor
        Serial.println("Predicciones:");
        for (int i = 0; i < NUM_CLASSES; i++) {
          float score = 0.0f;
          
          // Leer según el tipo de tensor
          if (output_tensor->type == kTfLiteFloat32) {
            score = output_tensor->data.f[i];
          } else if (output_tensor->type == kTfLiteInt8) {
            score = output_tensor->data.int8[i] / 127.0f;
          }
          
          Serial.printf("  %s: %.3f\n", commandNames[i], score);
          
          if (score > bestScore) {
            bestScore = score;
            bestCommand = i;
          }
        }
        
        // Activar LED si se detectó un comando
        if (bestCommand >= 0) {
          activateLED(bestCommand);
          lastDetectionTime = millis();
        } else {
          Serial.println("Comando no reconocido con suficiente confianza");
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