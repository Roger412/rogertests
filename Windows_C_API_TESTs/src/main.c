#include "unicorn.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main() {
    uint32_t deviceCount = 0;
    BOOL onlyPaired = TRUE;  // TRUE = solo emparejados, FALSE = escaneo profundo

    // 🔁 Primer llamada: solo obtener cuántos dispositivos hay
    int status = UNICORN_GetAvailableDevices(NULL, &deviceCount, onlyPaired);

    if (status != UNICORN_ERROR_SUCCESS) {
        printf("❌ Failed to scan for devices. Error code: %d\n", status);
        printf("📣 Error detail: %s\n", UNICORN_GetLastErrorText());
        return 1;
    }

    if (deviceCount == 0) {
        printf("⚠️  No Unicorn devices found.\n");
        return 1;
    }

    // ✅ Segunda llamada: reservar memoria y obtener seriales
    UNICORN_DEVICE_SERIAL* serials = (UNICORN_DEVICE_SERIAL*)malloc(sizeof(UNICORN_DEVICE_SERIAL) * deviceCount);
    if (serials == NULL) {
        printf("❌ Memory allocation failed.\n");
        return 1;
    }

    status = UNICORN_GetAvailableDevices(serials, &deviceCount, onlyPaired);
    if (status != UNICORN_ERROR_SUCCESS) {
        printf("❌ Failed to retrieve device serials. Error code: %d\n", status);
        printf("📣 Error detail: %s\n", UNICORN_GetLastErrorText());
        free(serials);
        return 1;
    }

    printf("✅ Found %u device(s):\n", deviceCount);
    for (uint32_t i = 0; i < deviceCount; ++i) {
        printf("   [%u] Serial: %s\n", i, serials[i]);
    }

    // 📡 Intentamos abrir el primer dispositivo
    UNICORN_HANDLE handle = 0;
    status = UNICORN_OpenDevice(serials[0], &handle);

    if (status == UNICORN_ERROR_SUCCESS) {
        printf("✅ Successfully connected to device: %s\n", serials[0]);

        // Aquí puedes hacer más cosas con el dispositivo...

        UNICORN_CloseDevice(&handle);
    } else {
        printf("❌ Failed to open device %s. Error code: %d\n", serials[0], status);
        printf("📣 Error detail: %s\n", UNICORN_GetLastErrorText());
    }

    // 🧼 Liberar memoria
    free(serials);

    return 0;
}
