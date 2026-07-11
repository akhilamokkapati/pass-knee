/*
 * knee_imu_serial.ino
 * PASS knee module — XIAO ESP32-C3 + 2x BNO085 -> serial packet stream.
 *
 * Emits one CSV line per sample in EXACTLY the contract that Python's
 * sources/serial_source.py parses (they are two halves of one tested contract):
 *
 *     seq,t_ms,knee_angle_deg,qtw,qtx,qty,qtz,qsw,qsx,qsy,qsz\n
 *
 *   seq            sample counter (uint)
 *   t_ms           millis() at emit (uint)
 *   knee_angle_deg ROUGH on-device angle  <-- CROSS-CHECK ONLY, not the truth
 *   qt{w,x,y,z}    thigh quaternion  (BNO085 game rotation vector, scalar-first)
 *   qs{w,x,y,z}    shank quaternion
 *
 * TRUTH LIVES IN PYTHON. The two raw quaternions are the source of truth: the
 * Python engine recomputes the knee angle from them via swing-twist (the
 * validated path). knee_angle_deg here is a deliberately rough total-rotation
 * estimate carried only so the host can cross-check firmware vs engine. Do NOT
 * grow this into the "real" angle on-device.
 *
 * ---------------------------------------------------------------------------
 * BRING-UP CHECKLIST (sensor day = flash + validate)
 * ---------------------------------------------------------------------------
 * 1. LIBRARY: install "SparkFun BNO080 Cortex Based IMU" via Arduino Library
 *    Manager (works with BNO080/085/086). Header included below.
 * 2. BOARD: Seeed XIAO ESP32-C3 (or S3 — same I2C pins). Select it in the IDE.
 * 3. POWER: BNO085 is 3.3V ONLY. Feed both breakouts from the XIAO 3V3 pin.
 *    Never 5V. Common ground.
 * 4. I2C WIRING (XIAO: SDA=D4/GPIO6, SCL=D5/GPIO7). Bus shared by both sensors:
 *      XIAO 3V3  -> both VIN/3V3
 *      XIAO GND  -> both GND
 *      XIAO D4   -> both SDA
 *      XIAO D5   -> both SCL
 * 5. MODE STRAPS on EACH breakout (force I2C):  PS0 -> GND,  PS1 -> GND.
 * 6. ADDRESS via ADO on EACH breakout (this is how two sensors share one bus):
 *      THIGH sensor: ADO -> GND  => I2C address 0x4A
 *      SHANK sensor: ADO -> 3V3  => I2C address 0x4B
 * 7. FLASH this sketch, open Serial Monitor at 115200.
 *    - Startup prints (all '#'-prefixed so the host parser ignores them):
 *          # PASS knee IMU bring-up
 *          # thigh BNO085 found at 0x4A        (or: NOT FOUND -> check ADO=GND, wiring)
 *          # shank BNO085 found at 0x4B        (or: NOT FOUND -> check ADO=3V3, wiring)
 *          # streaming: seq,t_ms,knee_angle_deg,qtw,qtx,qty,qtz,qsw,qsx,qsy,qsz
 *    - A HEALTHY data line looks like (quaternions unit-norm, ~4 sig figs):
 *          0,12,3.42,0.999877,0.006134,-0.010742,0.004150,0.965881,0.258789,0.012451,-0.001220
 *      Standing still: knee_angle_deg small and steady; both quats near-constant.
 *      Bend the knee: knee_angle_deg rises smoothly; quats change smoothly.
 *    - If a sensor is NOT FOUND: check its ADO strap, PS0/PS1=GND, 3V3, and SDA/SCL.
 * 8. HOST: python -c "from sources.serial_source import SerialSource;
 *    print(SerialSource(port='COM?').get_data(2).quat_thigh.shape)"  (flash+validate)
 *
 * ---------------------------------------------------------------------------
 * HARDWARE GOTCHAS baked into setup() below
 * ---------------------------------------------------------------------------
 *  - BNO085 CLOCK-STRETCHES on I2C. Run the bus at 100 kHz (Wire.setClock),
 *    higher rates are unreliable with this part.
 *  - Native USB CDC serial on the C3/S3 ignores the nominal baud; throughput is
 *    USB-speed, so 100 Hz of ~100-char lines streams fine. We still begin at
 *    115200 to match the host SerialSource default.
 */

#include <Wire.h>
#include "SparkFun_BNO080_Arduino_Library.h"

// ---- configuration ---------------------------------------------------------
static const uint8_t  THIGH_ADDR = 0x4A;   // ADO -> GND
static const uint8_t  SHANK_ADDR = 0x4B;   // ADO -> 3V3
static const uint32_t I2C_HZ     = 100000; // BNO085 clock-stretching: keep at 100 kHz
static const uint16_t REPORT_MS  = 10;     // BNO085 game rotation vector interval (~100 Hz)
static const uint32_t EMIT_MS    = 10;     // serial emit cadence (~100 Hz)

BNO080 thigh;
BNO080 shank;

// latest quaternion per segment (scalar-first); init to identity so the first
// lines are valid even before the first report arrives.
float tw = 1, tx = 0, ty = 0, tz = 0;
float sw = 1, sx = 0, sy = 0, sz = 0;

uint32_t seq = 0;
uint32_t lastEmit = 0;

// Rough on-device knee angle: total relative rotation between the two
// orientations = 2*acos(|dot(qt,qs)|). Mounting-dependent and not axis-isolated
// ON PURPOSE — it is only a cross-check; the Python engine does swing-twist.
float roughKneeAngleDeg() {
  float dot = tw * sw + tx * sx + ty * sy + tz * sz;
  dot = fabs(dot);
  if (dot > 1.0f) dot = 1.0f;
  return 2.0f * acos(dot) * 180.0f / PI;
}

void setup() {
  Serial.begin(115200);
  uint32_t t0 = millis();
  while (!Serial && (millis() - t0) < 3000) { /* wait up to 3 s for USB CDC */ }

  Wire.begin(D4, D5);        // XIAO ESP32-C3: SDA=D4 (GPIO6), SCL=D5 (GPIO7)
  Wire.setClock(I2C_HZ);     // 100 kHz — required for BNO085 clock stretching

  Serial.println("# PASS knee IMU bring-up");

  // Per-sensor diagnostics so day-one debugging is fast.
  if (thigh.begin(THIGH_ADDR, Wire)) {
    thigh.enableGameRotationVector(REPORT_MS);
    Serial.println("# thigh BNO085 found at 0x4A");
  } else {
    Serial.println("# thigh BNO085 NOT FOUND at 0x4A  (check ADO->GND, PS0/PS1->GND, 3V3, SDA/SCL)");
  }

  if (shank.begin(SHANK_ADDR, Wire)) {
    shank.enableGameRotationVector(REPORT_MS);
    Serial.println("# shank BNO085 found at 0x4B");
  } else {
    Serial.println("# shank BNO085 NOT FOUND at 0x4B  (check ADO->3V3, PS0/PS1->GND, 3V3, SDA/SCL)");
  }

  Serial.println("# streaming: seq,t_ms,knee_angle_deg,qtw,qtx,qty,qtz,qsw,qsx,qsy,qsz");
}

void loop() {
  // Cache the freshest quaternion from each sensor as reports arrive.
  if (thigh.dataAvailable()) {
    tw = thigh.getQuatReal(); tx = thigh.getQuatI();
    ty = thigh.getQuatJ();    tz = thigh.getQuatK();
  }
  if (shank.dataAvailable()) {
    sw = shank.getQuatReal(); sx = shank.getQuatI();
    sy = shank.getQuatJ();    sz = shank.getQuatK();
  }

  // Emit at a steady cadence using the latest cached values (decouples the
  // serial line rate from per-sensor report arrival).
  uint32_t now = millis();
  if (now - lastEmit >= EMIT_MS) {
    lastEmit = now;

    Serial.print(seq);            Serial.print(',');
    Serial.print(now);            Serial.print(',');
    Serial.print(roughKneeAngleDeg(), 2); Serial.print(',');
    Serial.print(tw, 6); Serial.print(',');
    Serial.print(tx, 6); Serial.print(',');
    Serial.print(ty, 6); Serial.print(',');
    Serial.print(tz, 6); Serial.print(',');
    Serial.print(sw, 6); Serial.print(',');
    Serial.print(sx, 6); Serial.print(',');
    Serial.print(sy, 6); Serial.print(',');
    Serial.println(sz, 6);

    seq++;
  }
}
