# 🔢 SYCL Vector Add (USM vs Buffers)

This sample demonstrates vector addition using SYCL with two memory models:
- **USM (Unified Shared Memory)**
- **Buffers + Accessors**

## 🧠 Key Difference (≤ 50 words)

**USM** uses raw pointers and requires manual memory management.  
**Buffers** abstract memory with automatic data movement and synchronization.  
USM is more explicit and low-level; Buffers are higher-level and safer.

---

## ⚙️ Build Instructions

### USM Version

```bash
cd vector-add
mkdir -p build && cd build
cmake .. -DUSM=1
make cpu-gpu
```
### Buffer Version
```bash
cd vector-add
mkdir -p build && cd build
cmake ..
make cpu-gpu
```