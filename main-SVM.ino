#include <math.h>
#include "svm_model.h"
#include <ESP32Servo.h>

// Create an instance of the SVM class
Eloquent::ML::Port::SVM svm;

// Definisikan pin RX dan TX untuk UART
#define RXD2 16
#define TXD2 17

// Jumlah channel dan ukuran set
const int numChannels = 8;
const int maxSets = 30;  // Jumlah data yang diperlukan untuk setiap channel
const int wlHistorySize = 10; // Ukuran untuk menyimpan riwayat WL

// Buffer untuk menyimpan data sementara dari setiap channel
int channelData[numChannels][maxSets];
int dataCount[numChannels] = {0};

// Variabel untuk menampilkan hasil per channel
float wlValues[numChannels];
float wlHistory[numChannels][wlHistorySize]; // Array untuk menyimpan riwayat WL
int wlIndex[numChannels] = {0}; // Indeks untuk riwayat WL
float wlNormalizedValues[numChannels]; // Array untuk menyimpan nilai normalisasi

// Nilai minimum dan maksimum untuk normalisasi min-max
const float minVal[numChannels] = {96.966019, 54.672330, 17.123786, 0.000000, 
                                   1.650485, 17.742718, 57.973301, 111.407767};
const float maxVal[numChannels] = {220.133495, 152.669903, 138.847087, 141.735437, 
                                   124.405340, 130.594660, 190.424757, 255.000000};

// Mean dan standar deviasi untuk StandardScaler
const float mean[numChannels] = {300.25909555, 248.62291738, 210.17885073, 203.84971098, 
                                 205.14858892, 218.27405644, 262.27541652, 316.52907174};
const float std_dev[numChannels] = {56.08416033, 52.10837585, 50.41390035, 52.3044522, 
                                    53.17310547, 54.22444168, 56.06630393, 60.29552283};

// Servo definitions
Servo servo1, servo2, servo3, servo4, servo5;
int servoPins[] = {9, 10, 11, 12, 13};  // Servo pins

// Function prototypes
void applyGaussianFilter(int data[], float result[], int data_size, float sigma, int kernel_size);
float calculateWaveformLength(float data[], int length);
float standardScale(float value, float mean, float std_dev);
float normalizeToCustomRange(float value, float min, float max);
int classifyWithSVM(float* input);
void moveServosToPosition(int pos1, int pos2, int pos3, int pos4, int pos5);
void performAction(int actionIndex);
void displayServoPositions();
Servo* getServo(int index);

// Inisialisasi komunikasi serial dan servo
void setup() {
    Serial.begin(115200);
    Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);
    Serial.println("Memulai pengumpulan data...");

    // Attach each servo to the corresponding pin
    servo1.attach(servoPins[0]);
    servo2.attach(servoPins[1]);
    servo3.attach(servoPins[2]);
    servo4.attach(servoPins[3]);
    servo5.attach(servoPins[4]);

    // Set servos to initial position (0 degrees)
    moveServosToPosition(0, 0, 0, 0, 0);
    delay(1000);  // Wait for 1 second
}

void loop() {
    
    // Data processing for SVM classification
    if (Serial2.available() >= numChannels) {
        // Ambil waktu awal loop
    unsigned long startTime = millis();
        for (int i = 0; i < numChannels; i++) {
            if (dataCount[i] < maxSets) {
                channelData[i][dataCount[i]] = Serial2.read();
                dataCount[i]++;
            }
        }

        bool allChannelsFull = true;
        for (int i = 0; i < numChannels; i++) {
            if (dataCount[i] < maxSets) {
                allChannelsFull = false;
                break;
            }
        }

        if (allChannelsFull) {
            for (int i = 0; i < numChannels; i++) {
                float filteredData[maxSets];
                applyGaussianFilter(channelData[i], filteredData, maxSets, 1.591, 11);
                wlValues[i] = calculateWaveformLength(filteredData, maxSets);

                wlHistory[i][wlIndex[i]] = wlValues[i];
                wlIndex[i] = (wlIndex[i] + 1) % wlHistorySize;

                Serial.print("Channel "); Serial.print(i + 1);
                Serial.print(" | WL: "); Serial.println(wlValues[i]);
            }

            for (int i = 0; i < numChannels; i++) {
                float wlScaled = standardScale(wlValues[i], mean[i], std_dev[i]);
                wlNormalizedValues[i] = normalizeToCustomRange(wlScaled, minVal[i], maxVal[i]);

                // Print the scaled value
                Serial.print("Channel "); Serial.print(i + 1);
                Serial.print(" | Nilai Scaled: "); Serial.println(wlScaled);
                
                Serial.print("Channel "); Serial.print(i + 1);
                Serial.print(" | Nilai Normalisasi: "); Serial.println(wlNormalizedValues[i]);
            }

            int classLabel = classifyWithSVM(wlNormalizedValues);
            Serial.print("Kelas Hasil Klasifikasi: "); Serial.println(classLabel);

            // Control servos based on classification
            performAction(classLabel);

            for (int i = 0; i < numChannels; i++) {
                dataCount[i] = 0;
            }
            unsigned long endTime = millis();
            unsigned long duration = endTime - startTime;  // Durasi dalam milidetik
                 // Tampilkan durasi loop di Serial Monitor
            Serial.print("Durasi waktu loop: ");
            Serial.print(duration);
            Serial.println(" ms");
            Serial.print("Total Maksimal Memori: ");
            Serial.println(ESP.getHeapSize());
            Serial.print("Sisa Memori Saat Penggunaan Maksimal: ");
            Serial.println(ESP.getMinFreeHeap());
            Serial.print("Sisa Memori Saat ini: ");
            Serial.println(ESP.getFreeHeap());
            Serial.print("Penggunaan Memori Saat Ini: ");
            Serial.println(ESP.getHeapSize() - ESP.getMinFreeHeap());
        }
    }

    delay(50); // Stability delay
}

// Function implementations

void applyGaussianFilter(int data[], float result[], int data_size, float sigma, int kernel_size) {
    float kernel[11];
    int pad = kernel_size / 2;
    // Fill kernel with Gaussian values and normalize
    float sum = 0.0;
    for (int i = 0; i < kernel_size; i++) {
        int x = i - pad;
        kernel[i] = exp(-(x * x) / (2 * sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < kernel_size; i++) {
        kernel[i] /= sum;
    }

    for (int i = 0; i < data_size; i++) {
        result[i] = 0.0;
        for (int j = -pad; j <= pad; j++) {
            int idx = i + j;
            if (idx < 0) idx = 0;
            if (idx >= data_size) idx = data_size - 1;
            result[i] += data[idx] * kernel[j + pad];
        }
    }
}

float calculateWaveformLength(float data[], int length) {
    float waveformLength = 0.0;
    for (int i = 1; i < length; i++) {
        waveformLength += abs(data[i] - data[i - 1]);
    }
    return waveformLength;
}

float standardScale(float value, float mean, float std_dev) {
    return (value - mean) / std_dev;
}

float normalizeToCustomRange(float value, float min, float max) {
    return (value - (-5)) / (5 - (-5)) * (max - min) + min;
}

int classifyWithSVM(float* input) {
    return svm.predict(input);
}

void moveServosToPosition(int pos1, int pos2, int pos3, int pos4, int pos5) {
    servo1.write(pos1);
    servo2.write(pos2);
    servo3.write(pos3);
    servo4.write(pos4);
    servo5.write(pos5);
    delay(1000);
    displayServoPositions();
}

void performAction(int actionIndex) {
    switch (actionIndex) {
        case 0: moveServosToPosition(0, 0, 0, 0, 0); break;
        case 1: moveServosToPosition(180, 180, 180, 180, 180); break;
        case 2: moveServosToPosition(180, 145, 0, 0, 0); break;
        case 3: moveServosToPosition(180, 0, 180, 180, 160); break;
        case 4: moveServosToPosition(0, 160, 180, 180, 160); break;
        case 5: moveServosToPosition(180, 0, 0, 180, 160); break;
        case 6: moveServosToPosition(180, 0, 0, 0, 160); break;
        case 7: moveServosToPosition(180, 0, 0, 0, 0); break;
        case 8: moveServosToPosition(160, 140, 100, 180, 160); break;
        case 9: moveServosToPosition(160, 140, 110, 180, 120); break;
        default: moveServosToPosition(0, 0, 0, 0, 0); break;
    }
}

void displayServoPositions() {
    Serial.print("Servo 1 Posisi: "); Serial.println(servo1.read());
    Serial.print("Servo 2 Posisi: "); Serial.println(servo2.read());
    Serial.print("Servo 3 Posisi: "); Serial.println(servo3.read());
    Serial.print("Servo 4 Posisi: "); Serial.println(servo4.read());
    Serial.print("Servo 5 Posisi: "); Serial.println(servo5.read());
}

Servo* getServo(int index) {
    switch (index) {
        case 0: return &servo1;
        case 1: return &servo2;
        case 2: return &servo3;
        case 3: return &servo4;
        case 4: return &servo5;
        default: return nullptr;
    }
}
