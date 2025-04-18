#include "unicorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DURATION_SECONDS 5  // Cambia la duración aquí
#define FRAME_LENGTH 1
#define FILENAME "data.csv"

int main() {
    uint32_t deviceCount = 0;
    BOOL onlyPaired = TRUE;
    int status = UNICORN_GetAvailableDevices(NULL, &deviceCount, onlyPaired);

    if (status != UNICORN_ERROR_SUCCESS || deviceCount == 0) {
        printf("❌ No devices found. Error %d: %s\n", status, UNICORN_GetLastErrorText());
        return 1;
    }

    UNICORN_DEVICE_SERIAL* serials = (UNICORN_DEVICE_SERIAL*)malloc(sizeof(UNICORN_DEVICE_SERIAL) * deviceCount);
    status = UNICORN_GetAvailableDevices(serials, &deviceCount, onlyPaired);
    if (status != UNICORN_ERROR_SUCCESS) {
        printf("❌ Failed to get device serials. %s\n", UNICORN_GetLastErrorText());
        free(serials);
        return 1;
    }

    UNICORN_HANDLE handle = 0;
    status = UNICORN_OpenDevice(serials[0], &handle);
    free(serials);

    if (status != UNICORN_ERROR_SUCCESS) {
        printf("❌ Failed to open device. %s\n", UNICORN_GetLastErrorText());
        return 1;
    }

    // Obtener número de canales
    uint32_t numChannels = 0;
    status = UNICORN_GetNumberOfAcquiredChannels(handle, &numChannels);
    if (status != UNICORN_ERROR_SUCCESS) {
        printf("❌ Failed to get number of channels. %s\n", UNICORN_GetLastErrorText());
        UNICORN_CloseDevice(&handle);
        return 1;
    }

    uint32_t bufferSize = numChannels * FRAME_LENGTH;
    float* buffer = (float*)malloc(sizeof(float) * bufferSize);

    if (buffer == NULL) {
        printf("❌ Memory allocation failed.\n");
        UNICORN_CloseDevice(&handle);
        return 1;
    }

    FILE* file = fopen(FILENAME, "w");
    if (!file) {
        printf("❌ Could not open output file.\n");
        free(buffer);
        UNICORN_CloseDevice(&handle);
        return 1;
    }

    UNICORN_AMPLIFIER_CONFIGURATION config;
    status = UNICORN_GetConfiguration(handle, &config);
    if (status != UNICORN_ERROR_SUCCESS) {
        printf("❌ Failed to get configuration. %s\n", UNICORN_GetLastErrorText());
        UNICORN_CloseDevice(&handle);
        return 1;
    }

    // 📝 Guardar cabecera con nombres reales
    for (uint32_t i = 0; i < numChannels; ++i) {
        fprintf(file, "%s%s", config.Channels[i].name, (i == numChannels - 1) ? "\n" : ",");
    }

    // Iniciar adquisición
    status = UNICORN_StartAcquisition(handle, FALSE);
    if (status != UNICORN_ERROR_SUCCESS) {
        printf("❌ Failed to start acquisition. %s\n", UNICORN_GetLastErrorText());
        fclose(file);
        free(buffer);
        UNICORN_CloseDevice(&handle);
        return 1;
    }

    printf("🔴 Acquiring for %d seconds...\n", DURATION_SECONDS);
    uint32_t totalFrames = UNICORN_SAMPLING_RATE * DURATION_SECONDS;

    for (uint32_t i = 0; i < totalFrames; ++i) {
        status = UNICORN_GetData(handle, FRAME_LENGTH, buffer, bufferSize * sizeof(float));
        if (status != UNICORN_ERROR_SUCCESS) {
            printf("❌ Data acquisition failed. %s\n", UNICORN_GetLastErrorText());
            break;
        }

        // Escribir al archivo
        for (uint32_t j = 0; j < bufferSize; ++j) {
            fprintf(file, "%.6f%s", buffer[j], (j == bufferSize - 1) ? "\n" : ",");
        }
    }

    UNICORN_StopAcquisition(handle);
    UNICORN_CloseDevice(&handle);
    fclose(file);
    free(buffer);

    printf("✅ Acquisition complete! Data saved to: %s\n", FILENAME);
    return 0;
}
