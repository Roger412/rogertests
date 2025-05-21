# TO DO

- Check if video works in station mode ‼️


# 📡 Connecting Tello Drone via Wi-Fi Hotspot on Linux

This guide explains how to connect your DJI Tello drone to a computer-hosted Wi-Fi hotspot using [linux-wifi-hotspot](https://github.com/lakinduakash/linux-wifi-hotspot), and how to control it using the `djitellopy` Python library.

---

## 🔧 Requirements

- A Linux machine (e.g., Ubuntu)
- Installed [`linux-wifi-hotspot`](https://github.com/lakinduakash/linux-wifi-hotspot)
- Python 3 + `djitellopy` (`pip install djitellopy`)
- DJI Tello drone (standard or EDU)

---

## 📶 Step 1: Create a Wi-Fi Hotspot

Launch `linux-wifi-hotspot` with the command `wihotspot` and configure it as follows:

| Field               | Value         |
|--------------------|---------------|
| SSID               | `rogernet`    |
| Password           | `########`   |
| Wifi Interface     | `wlp0s20f3`   |
| Internet Interface | `wlp0s20f3`   |
| Frequency Band     | `2.4 GHz`     |
| Gateway            | `192.168.12.1`|

> ✅ **NOTE:** Tello only supports **2.4GHz** Wi-Fi.

Click **Create Hotspot**.

---

## ✈️ Step 2: Send Wi-Fi Credentials to the Drone

1. Connect your computer to the default Tello Wi-Fi (`TELLO-XXXXXX`).
2. Run this Python code to tell the drone to connect to your hotspot:

```python
from djitellopy import Tello
import time

tello = Tello()
tello.connect()

# Send credentials to join your hotspot
tello.send_command_with_return("ap rogernet ########")
print("✅ Sent Wi-Fi credentials")

# Allow the drone to reboot
time.sleep(5)
tello.end()
```

3. **Power cycle the drone** if it doesn’t automatically reboot.

---

## 🔍 Step 3: Get Drone's IP Address

Once the drone reboots, it will try to connect to `rogernet`.

- Go back to the **WiFi Hotspot** GUI.
- Scroll to **Connected devices** (refresh if needed).
- Look for an IP like `192.168.12.6` — that is your drone.

---

## 🧠 Step 4: Connect and Control the Drone

Use the discovered IP in your script:

```python
from djitellopy import Tello

# Replace with IP discovered in step 3
tello = Tello(host="192.168.12.6")
tello.connect()

print("Battery level:", tello.get_battery())
tello.takeoff()
tello.land()
```

---

## 📝 Notes

- Stay connected to `rogernet` while communicating with the drone.
- **Video streaming (`streamon`) may not work in Station mode**.
- This works for both Tello EDU and Standard (EDU is more stable in STA mode).

---

## 🧪 Optional: Scan Network with nmap

If you don’t see your drone in the hotspot’s GUI:

```bash
sudo apt install nmap
sudo nmap -sn 192.168.12.0/24
```

Look for a MAC address that starts with `60:60:1F` (Ryze Tech).

---

✅ You’re all set to fly your Tello over your own network!
