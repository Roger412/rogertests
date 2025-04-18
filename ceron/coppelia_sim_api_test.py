from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time

client = RemoteAPIClient()
sim = client.getObject('sim')

left_motor = sim.getObject('/PioneerP3DX/leftMotor')
right_motor = sim.getObject('/PioneerP3DX/rightMotor')

sim.setJointTargetVelocity(left_motor, 2.0)
sim.setJointTargetVelocity(right_motor, 2.0)

time.sleep(2)

sim.setJointTargetVelocity(left_motor, 0.0)
sim.setJointTargetVelocity(right_motor, 0.0)
