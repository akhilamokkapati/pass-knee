// PID learning test — TSA tension-control loop, NO real hardware yet.
// The "plant" below is a fake first-order model
// standing in for (motor + driver + string mechanics), 
//
// Once a strain gauge and a motor driver are wired and chosen, two things
// change here and nothing else: readSimulatedTension() becomes a real
// analogRead()-based conversion, and setMotor() gets real GPIO/PWM calls for
// whichever driver (DRV8833/DRV8874) is picked.

#include <Arduino.h>
#include <PID_v1.h>

// ---- PID wiring -----------------------------------------------------------
// The library reads/writes these three directly 
// — we update Input ourselves each loop, the library
// overwrites Output when Compute() actually fires.

double Setpoint; // target tension (arbitrary units for this mock — think "N")
double Input;    // measured tension 
double Output;   // signed motor command: + = twist (add tension), - = untwist

// Starting gains — arbitrary for this learning pass, 
double Kp = 2.0, Ki = 0.5, Kd = 0.1;

PID myPID(&Input, &Output, &Setpoint, Kp, Ki, Kd, DIRECT);

// ---- Mock plant -------------------------------------------------------------
// Stands in for "motor + driver + TSA mechanics" until real hardware exists.
// Tension drifts toward a level proportional to accumulated motor command,
// with drag pulling it back toward zero — enough first-order lag/decay to
// see a real step response instead of an instant jump.
double simulatedTension = 0.0;
// First-order lag: tension chases (PLANT_K * output) with time constant
// PLANT_TAU_S. PLANT_K is chosen so max output (255) settles at 150 -
// comfortably above the 100 setpoint, so the loop can actually reach
// steady state instead of saturating forever (that's a different failure
// mode - integrator windup - worth causing on purpose later, not by
// accident of an unreachable target).
const double PLANT_K = 150.0 / 255.0;
const double PLANT_TAU_S = 1.0; // seconds - how "sluggish" the fake plant is

unsigned long lastPlantUpdateUs = 0;

void updateSimulatedPlant(double motorCommand) {
  unsigned long nowUs = micros();
  double dtSeconds = (nowUs - lastPlantUpdateUs) / 1000000.0;
  lastPlantUpdateUs = nowUs;

  double targetTension = PLANT_K * motorCommand;
  // Euler step of dTension/dt = (target - tension) / tau, scaled by REAL
  // elapsed time (not loop() iteration count), so the ramp plays out over
  // ~PLANT_TAU_S seconds on the clock no matter how fast loop() spins.
  simulatedTension += (targetTension - simulatedTension) * (dtSeconds / PLANT_TAU_S);
  if (simulatedTension < 0) simulatedTension = 0; // string can't push, only pull
}

// ---- Motor output seam ------------------------------------------------------
// Driver-agnostic on purpose: DRV8833 vs DRV8874 isn't decided yet 
// This just logs what it WOULD do; real IN1/IN2 + PWM calls drop in here
// once a driver is chosen, without touching the PID logic above.
void setMotor(double signedPwm) {
  int magnitude = (int)abs(signedPwm);
  if (signedPwm > 1.0) {
    Serial.print(F("  motor: TWIST   pwm="));
    Serial.println(magnitude);
  } else if (signedPwm < -1.0) {
    Serial.print(F("  motor: UNTWIST pwm="));
    Serial.println(magnitude);
  } else {
    Serial.println(F("  motor: HOLD"));
  }
}

// ---- Disturbance injection ---------------------------------------------------
// One-shot simulated "patient moves the knee" event, to make Phase 2
// visible in the trace, not just Phase 1
// (ramp to setpoint).
const unsigned long DISTURBANCE_AT_MS = 6000;
bool disturbanceApplied = false;

unsigned long lastPrintMs = 0;
const unsigned long PRINT_INTERVAL_MS = 100;

void setup() {
  Serial.begin(115200);
  delay(500);

  Setpoint = 100.0; // arbitrary target tension for this learning run

  // Default output range is 0-255 (assumes a unidirectional motor). Widening
  // to a negative minimum is what makes the SAME PID instance produce both
  // twist and untwist commands
 
  myPID.SetOutputLimits(-255, 255);
  myPID.SetMode(AUTOMATIC); // arms the loop; bumpless (no kick on startup)

  lastPlantUpdateUs = micros(); // clean dt on the very first plant update

  Serial.println(F("t_ms,setpoint,tension,motor_command"));
}

void loop() {
  Input = simulatedTension;

  // Safe to call every iteration — Compute() internally no-ops until its own
  // sample-time interval (default 100 ms) has elapsed.
  myPID.Compute();

  updateSimulatedPlant(Output);
  setMotor(Output);

  // Simulate the patient perturbing tension partway through the run, once.
  if (!disturbanceApplied && millis() >= DISTURBANCE_AT_MS) {
    simulatedTension = max(0.0, simulatedTension - 40.0);
    disturbanceApplied = true;
    Serial.println(F("# disturbance applied (simulated patient motion)"));
  }

  unsigned long now = millis();
  if (now - lastPrintMs >= PRINT_INTERVAL_MS) {
    lastPrintMs = now;
    Serial.print(now);
    Serial.print(',');
    Serial.print(Setpoint);
    Serial.print(',');
    Serial.print(simulatedTension);
    Serial.print(',');
    Serial.println(Output);
  }
}
