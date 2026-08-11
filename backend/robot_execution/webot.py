import math
import threading
import time
from backend.robot_execution.robot_interface import RobotInterface

class WebotDriver(RobotInterface):
    def __init__(self, robot):
        self.robot = robot
        self.time_step = int(self.robot.getBasicTimeStep())
        # The Webots controller API is not thread-safe, and two threads drive it
        # concurrently: DetectorService pulls camera frames while the Executor
        # agent runs motion loops. Serialise every simulation step through this
        # lock. It is deliberately fine-grained (one step at a time, never a
        # whole motion loop) so the camera keeps producing frames while the
        # robot manoeuvres.
        self.step_lock = threading.RLock()
        
        # Sensors
        self.camera = None
        self.distance_sensor = None
        self.front_distance_sensors: list = []
        self.gps = None
        self.compass = None
        
        # Actuators
        self.left_motor = None
        self.right_motor = None
        self.arm_motor = None
        self.gripper_motor = None

    def initialize_devices(self):
        """Connects to the simulator hardware/devices."""
        # 1. Sensors
        # Note: If your world definition doesn't include a camera, 
        # this safely catches the None without crashing.
        self.camera = self.robot.getDevice("camera") or self.robot.getDevice("khepera3_gripper_camera")
        if self.camera:
            self.camera.enable(self.time_step)

        # ds3/ds4 are the front-centre pair (proto translation x=+0.0656,
        # y=-+0.0155). ds0 sits at (-0.0404, +0.0496) — rear-left — so reading it
        # as "distance to front" reports empty space whenever the robot is
        # actually facing its target. The legacy C controller only ever used ds0
        # as "sensor index 0" while sweeping the whole ring.
        self.front_distance_sensors = []
        for sensor_name in ("ds3", "ds4"):
            sensor = self.robot.getDevice(sensor_name)
            if sensor:
                sensor.enable(self.time_step)
                self.front_distance_sensors.append(sensor)

        if self.front_distance_sensors:
            self.distance_sensor = self.front_distance_sensors[0]
        else:  # pragma: no cover - only if the robot lacks the standard ring
            self.distance_sensor = self.robot.getDevice("ds0")
            if self.distance_sensor:
                self.distance_sensor.enable(self.time_step)
                self.front_distance_sensors = [self.distance_sensor]

        # Tracking sensors (Safely handled if not present in your specific .wbt world file)
        self.gps = self.robot.getDevice("gps")
        if self.gps:
            self.gps.enable(self.time_step)

        self.compass = self.robot.getDevice("compass")
        if self.compass:
            self.compass.enable(self.time_step)

        # 2. Wheel Actuators (Velocity Control)
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        
        if self.left_motor and self.right_motor:
            self.left_motor.setPosition(float('inf'))
            self.right_motor.setPosition(float('inf'))
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

        # 3. Gripper Actuators (Position Control)
        # exact strings extracted from the provided C controller code
        self.arm_motor = self.robot.getDevice("horizontal_motor")
        self.gripper_motor = self.robot.getDevice("finger_motor::left")

    def step(self, duration_ms: int | None = None) -> int:
        """Advances the simulation by one step under the shared lock."""
        with self.step_lock:
            return self.robot.step(int(duration_ms) if duration_ms else self.time_step)

    def read_camera_image(self):
        """Steps the sim and grabs a camera frame as one atomic operation.

        Returns ``(raw_image, width, height)``, or ``(None, 0, 0)`` when the
        camera is absent or the simulation has ended.
        """
        if not self.camera:
            return None, 0, 0

        with self.step_lock:
            if self.robot.step(self.time_step) == -1:
                return None, 0, 0
            return (
                self.camera.getImage(),
                self.camera.getWidth(),
                self.camera.getHeight(),
            )

    def get_camera_width(self) -> int:
        if self.camera:
            return self.camera.getWidth()
        return 640  # Sensible default fallback

    def get_distance_to_front(self) -> float:
        """Distance to the nearest obstacle ahead, **in metres**.

        ``DistanceSensor.getValue()`` returns a raw lookup-table reading, not a
        length: on the Khepera3's infrared sensors that is 3983 at contact down
        to 7 at the ~0.45 m range limit — i.e. larger means *closer*. Callers
        (``ExecutorAgent._get_object`` → ``move_forward``) treat the result as
        metres, so returning the raw value commanded drives thousands of times
        too long. Invert the sensor's own lookup table instead of hard-coding a
        conversion, so this stays correct for any sensor or robot.

        Returns 0.0 when nothing is within range, which callers already treat
        as "no reachable target".
        """
        sensors = self.front_distance_sensors or (
            [self.distance_sensor] if self.distance_sensor else []
        )
        if not sensors:
            return 0.0

        # Nearest obstacle seen by either front sensor.
        readings = [
            metres
            for metres in (self._sensor_metres(sensor) for sensor in sensors)
            if metres > 0.0
        ]
        return min(readings) if readings else 0.0

    def _sensor_metres(self, sensor) -> float:
        """Convert one distance sensor's raw reading into metres."""
        raw = sensor.getValue()
        table = self._distance_lookup(sensor)
        if not table:
            return float(raw)

        # table is [(distance, raw), ...] sorted by increasing distance and,
        # for these IR sensors, decreasing raw value.
        if raw >= table[0][1]:
            return float(table[0][0])
        if raw <= table[-1][1]:
            # Below the weakest reading: nothing within usable range.
            return 0.0

        for (d_near, r_near), (d_far, r_far) in zip(table, table[1:]):
            if r_far <= raw <= r_near:
                span = r_near - r_far
                if span <= 0:
                    return float(d_near)
                ratio = (r_near - raw) / span
                return float(d_near + ratio * (d_far - d_near))

        return 0.0

    def _distance_lookup(self, sensor) -> list[tuple[float, float]]:
        """Cached [(distance_m, raw_value)] pairs for one sensor's lookup table."""
        cache = getattr(self, "_distance_lookup_cache", None)
        if cache is None:
            cache = {}
            self._distance_lookup_cache = cache

        key = id(sensor)
        if key in cache:
            return cache[key]

        pairs: list[tuple[float, float]] = []
        try:
            # Webots returns a flat [distance, value, noise, ...] triplet list.
            flat = sensor.getLookupTable()
            for i in range(0, len(flat) - 2, 3):
                pairs.append((float(flat[i]), float(flat[i + 1])))
            pairs.sort(key=lambda p: p[0])
        except Exception as exc:  # pragma: no cover - depends on the Webots build
            print(f"[WebotDriver] Could not read distance-sensor lookup table: {exc}")
            pairs = []

        cache[key] = pairs
        return pairs

    def get_position(self) -> tuple[float, float]:
        """Returns the current (x, y) coordinates of the robot from the GPS."""
        if self.gps:
            values = self.gps.getValues()
            return (values[0], values[1])
        return (0.0, 0.0)

    def get_angle(self) -> float:
        """Returns the current orientation/heading angle of the robot in radians using the Compass."""
        if self.compass:
            north = self.compass.getValues()
            return math.atan2(north[1], north[0])
        return 0.0

    def move_forward(self, distance: float):
        """Drives forward at a set velocity and blocks until distance is covered."""
        if not self.left_motor or not self.right_motor:
            return
            
        speed = 12.8  # Using the baseline operational speed from your C code
        self.left_motor.setVelocity(speed)
        self.right_motor.setVelocity(speed)
        
        duration = distance / 0.2 
        start_time = self.robot.getTime()
        
        while self.robot.getTime() - start_time < duration:
            if self.step() == -1:
                break
        self.stop()

    def move(self, linear_v: float, angular_v: float):
        """Set non-blocking differential-drive velocities for the RL bridge.

        ``WebotsBridge`` advances simulation time separately, so this method
        deliberately does not call ``robot.step`` or block.
        """
        if not self.left_motor or not self.right_motor:
            return

        # The constants match the existing driver speed scale while providing
        # a stable linear/angular command adapter.
        linear_speed = float(linear_v) * 12.8 / 0.2
        turn_speed = float(angular_v) * 2.4 / 0.5
        self.left_motor.setVelocity(linear_speed - turn_speed)
        self.right_motor.setVelocity(linear_speed + turn_speed)

    def step_simulation(self, duration_ms: int = 100):
        """Advance Webots once using the requested RL control interval."""
        return self.step(duration_ms)
        
    def turn(self, angle: float):
        """Rotates in place. Positive angle = Right, Negative = Left."""
        if not self.left_motor or not self.right_motor:
            return
            
        speed = 2.4  # Matches your C source baseline rotation speed
        if angle > 0:
            self.left_motor.setVelocity(speed)
            self.right_motor.setVelocity(-speed)
        else:
            self.left_motor.setVelocity(-speed)
            self.right_motor.setVelocity(speed)
            
        duration = abs(angle) * 0.012  
        start_time = self.robot.getTime()
        
        while self.robot.getTime() - start_time < duration:
            if self.step() == -1:
                break
        self.stop()
        
    def stop(self):
        """Stops wheel velocities completely."""
        if self.left_motor and self.right_motor:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

    def raise_arm(self):
        """Uses position control to lift the gripper assembly upwards."""
        if self.arm_motor:
            self.arm_motor.setVelocity(2.0)
            self.arm_motor.setPosition(0.0)  
            self._wait_for_actuator(1.0)      

    def lower_arm(self):
        """Uses position control to lower the arm down near the cylinder."""
        if self.arm_motor:
            self.arm_motor.setVelocity(2.0)
            self.arm_motor.setPosition(-3.0)  # Value pulled directly from line 118 of C source
            self._wait_for_actuator(1.0)
        
    def grab_item(self):
        """Closes the gripper claws inwards."""
        if self.gripper_motor:
            self.gripper_motor.setVelocity(2.0)
            self.gripper_motor.setPosition(0.42)  # Value pulled directly from line 122 of C source
            self._wait_for_actuator(1.0)
        
    def release_item(self):
        """Opens the gripper claws wide."""
        if self.gripper_motor:
            self.gripper_motor.setVelocity(2.0)
            self.gripper_motor.setPosition(0.0) 
            self._wait_for_actuator(1.0)

    def _wait_for_actuator(self, duration: float):
        """Helper to pause execution and let mechanical parts finish moving."""
        start = self.robot.getTime()
        while self.robot.getTime() - start < duration:
            if self.step() == -1:
                break
