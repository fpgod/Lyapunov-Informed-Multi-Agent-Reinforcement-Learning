
from controller import Robot


class EpuckRobot():
    def __init__(self):
        super().__init__()
        self.robot = Robot()
        #通过最后一个epuck的名称得到epuck总数，self.robot.getName():e-puck[]
        # print(self.robot.getName())
        self.robot_num = int(self.robot.getName()[-1])

        self.timestep = int(self.robot.getBasicTimeStep())

        self.emitter, self.receiver = self.initialize_comms()
        # 初始化emitter和receiver
        self.left_motor, self.right_motor = None, None
        self.ds = []
        for i in range(8):
            self.ds.append(self.robot.getDevice('ps' + str(i)))
            self.ds[i].enable(self.timestep)
        # ds[i]中存储第i个距离传感器的数值
        self.setup_motors()
        # 初始化电机



    def initialize_comms(self, emitter_name="emitter", receiver_name="receiver"):
        emitter = self.robot.getDevice(emitter_name)
        receiver = self.robot.getDevice(receiver_name)
        receiver.enable(self.timestep)
        return emitter, receiver

    def setup_motors(self):
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def create_message(self):
        """

        :return:
        """
        message = []
        for i in range(8):
            message.append(self.ds[i].getValue())
        return message

    def handle_receiver(self):
        while self.receiver.getQueueLength() > 0:
            message = self.receiver.getData().decode("utf-8")
            message = message.split(",")
            self.use_message_data(message)
            self.receiver.nextPacket()

    def use_message_data(self, message):
        self.setup_motors()
        action1 = float(message[0]) * 7.536
        action2 = float(message[1]) * 7.536
        self.left_motor.setVelocity(action1)
        self.right_motor.setVelocity(action2)

    def handle_emitter(self):
        data = self.create_message()
        string_message = ""

        if type(data) is list:
            string_message = ",".join(map(str, data))
        elif type(data) is str:
            string_message = data
        else:
            raise TypeError(
                "message must be either a comma-separater string or a 1D list")

        string_message = string_message.encode("utf-8")
        self.emitter.send(string_message)

    def run(self):
        while self.robot.step(self.timestep) != -1:
            self.handle_receiver()
            self.handle_emitter()


robot_controller = EpuckRobot()
robot_controller.run()
