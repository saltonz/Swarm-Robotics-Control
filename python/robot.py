import numpy as np
import pybullet as p
import itertools
import math

class Robot():
    """ 
    The class is the interface to a single robot
    """
    def __init__(self, init_pos, robot_id, dt):
        self.id = robot_id
        self.dt = dt
        self.pybullet_id = p.loadSDF("../models/robot.sdf")[0]
        self.joint_ids = list(range(p.getNumJoints(self.pybullet_id)))
        self.initial_position = init_pos
        self.reset()

        # No friction between bbody and surface.
        p.changeDynamics(self.pybullet_id, -1, lateralFriction=5., rollingFriction=0.)

        # Friction between joint links and surface.
        for i in range(p.getNumJoints(self.pybullet_id)):
            p.changeDynamics(self.pybullet_id, i, lateralFriction=5., rollingFriction=0.)
            
        self.messages_received = []
        self.messages_to_send = []
        self.neighbors = []
        self.step1_ref = {0: [1.5, 0],
                          1: [1.5, 1],
                          2: [2, 0],
                          3: [2, 1],
                          4: [2.5, 0],
                          5: [2.5, 1]}
        self.step2_topo = [2, 3, 4, 5, 1]
        # self.virtual_leader_pos = [1.5, 0]
        self.virtual_leader_pos = [2, 4]
        self.step6_topo = [2, 5, 4, 0, 7, 3]
        self.virtual_leader2_pos = [4, 2]
        self.step11_topo = [7, 4, 0, 1, 2, 3]
        self.step13_ref = {0: [1.65, -0.5],
                           1: [0.5, 0],
                           2: [1.35, -0.5],
                           3: [1.35, 0.5],
                           4: [2.5, 0],
                           5: [1.65, 0.5]}

        self.time = 0
        self.dt = 1. / 60.
        self.step = 0
        self.next_step = 0
        

    def reset(self):
        """
        Moves the robot back to its initial position 
        """
        p.resetBasePositionAndOrientation(self.pybullet_id, self.initial_position, (0., 0., 0., 1.))
            
    def set_wheel_velocity(self, vel):
        """ 
        Sets the wheel velocity,expects an array containing two numbers (left and right wheel vel) 
        """
        assert len(vel) == 2, "Expect velocity to be array of size two"
        p.setJointMotorControlArray(self.pybullet_id, self.joint_ids, p.VELOCITY_CONTROL,
            targetVelocities=vel)

    def get_pos_and_orientation(self):
        """
        Returns the position and orientation (as Yaw angle) of the robot.
        """
        pos, rot = p.getBasePositionAndOrientation(self.pybullet_id)
        euler = p.getEulerFromQuaternion(rot)
        return np.array(pos), euler[2]
    
    def get_messages(self):
        """
        returns a list of received messages, each element of the list is a tuple (a,b)
        where a= id of the sending robot and b= message (can be any object, list, etc chosen by user)
        Note that the message will only be received if the robot is a neighbor (i.e. is close enough)
        """
        return self.messages_received
        
    def send_message(self, robot_id, message):
        """
        sends a message to robot with id number robot_id, the message can be any object, list, etc
        """
        self.messages_to_send.append([robot_id, message])
        
    def get_neighbors(self):
        """
        returns a list of neighbors (i.e. robots within 2m distance) to which messages can be sent
        """
        return self.neighbors

    def compute_and_set_velocity(self, rot, dx, dy, gain):
        vel_norm = np.linalg.norm([dx, dy])  # norm of desired velocity
        if vel_norm < 0.01:
            vel_norm = 0.01
        des_theta = np.arctan2(dy / vel_norm, dx / vel_norm)
        right_wheel = np.sin(des_theta - rot) * vel_norm + np.cos(des_theta - rot) * vel_norm
        left_wheel = -np.sin(des_theta - rot) * vel_norm + np.cos(des_theta - rot) * vel_norm
        right_wheel *= gain
        left_wheel *= gain

        self.set_wheel_velocity([left_wheel, right_wheel])


    @staticmethod
    def compute_potential_field_gradient(distance, alpha, d_0, d_1):
        if 0 < distance < d_1:
            return alpha * (1.0 / distance - d_0 / (distance * distance))
        else:
            return 0

    @staticmethod
    def compute_repulsive_force_gradient(distance, alpha, d_0):
        if 0 < distance < d_0:
            return alpha * (-1.0 / ((distance + 0.5) * (distance + 0.5)))
        else:
            return 0

    
    def compute_controller(self):
        """ 
        function that will be called each control cycle which implements the control law
        TO BE MODIFIED
        
        we expect this function to read sensors (built-in functions from the class)
        and at the end to call set_wheel_velocity to set the appropriate velocity of the robots
        """
        neig = self.get_neighbors()
        messages = self.get_messages()
        pos, rot = self.get_pos_and_orientation()

        to_send = {"pos": pos,
                   "next_step": self.next_step}
        # send message of positions to all neighbors indicating our position
        for n in neig:
            self.send_message(n, to_send)
        leader_distance = 0.
        if self.step == 0:
            # square formation with translation invariant case control law

            # check if we received the position of our neighbors and compute desired change in position
            # as a function of the neighbors (message is composed of [neighbors id, position])
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dx += m[1]["pos"][0] - pos[0] - (self.step1_ref[m[0]][0] - self.step1_ref[self.id][0])
                    dy += m[1]["pos"][1] - pos[1] - (self.step1_ref[m[0]][1] - self.step1_ref[self.id][1])
                # integrate
                des_pos_x = pos[0] + self.dt * dx
                des_pos_y = pos[1] + self.dt * dy

                self.compute_and_set_velocity(rot, dx, dy, 10)

        elif self.step == 1:
            # move the swarm out of the room in a line
            if self.id == 5:
                dx = 0
                dy = 0
                if pos[0] < 2.5:
                    dx = 1
                    dy = 0
                elif pos[1] < 5.5:
                    dx = 0
                    dy = 1
                self.compute_and_set_velocity(rot, dx, dy, 7)
            else:
                dx = 0.
                dy = 0.
                if messages:
                    for m in messages:
                        dif_x = m[1]["pos"][0] - pos[0]
                        dif_y = m[1]["pos"][1] - pos[1]
                        if m[0] == self.step2_topo[self.id]:
                            dx += dif_x
                            dy += dif_y
                        distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                        repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.2, 0.3)
                        vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                        if vel_norm < 0.01:
                            vel_norm = 0.01
                        rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                        dx += repulsive_gradient * np.cos(rep_theta)
                        dy += repulsive_gradient * np.sin(rep_theta)

                self.compute_and_set_velocity(rot, dx, dy, 10)

        elif self.step == 2:
            # circle around the purple ball with radius = 0.5
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 1)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)


            # calculate the impact of virtual leader
            dif_leader_x = self.virtual_leader_pos[0] - pos[0]
            dif_leader_y = self.virtual_leader_pos[1] - pos[1]
            leader_distance = math.sqrt(dif_leader_x * dif_leader_x + dif_leader_y * dif_leader_y)
            sum_gradient = self.compute_potential_field_gradient(leader_distance, 5, 0.5, 5)
            vel_norm = np.linalg.norm([dif_leader_x, dif_leader_y])  # norm of
            if vel_norm < 0.01:
                vel_norm = 0.01
            vir_theta = np.arctan2(dif_leader_y / vel_norm, dif_leader_x / vel_norm)
            dx += sum_gradient * np.cos(vir_theta)
            dy += sum_gradient * np.sin(vir_theta)
            self.compute_and_set_velocity(rot, dx, dy, 7)

        elif self.step == 3:
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 1)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)


            # calculate the impact of virtual leader
            dif_leader_x = self.virtual_leader_pos[0] - pos[0]
            dif_leader_y = self.virtual_leader_pos[1] - pos[1]
            leader_distance = math.sqrt(dif_leader_x * dif_leader_x + dif_leader_y * dif_leader_y)
            sum_gradient = self.compute_potential_field_gradient(leader_distance, 5, 0.35, 5)
            vel_norm = np.linalg.norm([dif_leader_x, dif_leader_y])  # norm of
            if vel_norm < 0.01:
                vel_norm = 0.01
            vir_theta = np.arctan2(dif_leader_y / vel_norm, dif_leader_x / vel_norm)
            dx += sum_gradient * np.cos(vir_theta)
            dy += sum_gradient * np.sin(vir_theta)
            self.compute_and_set_velocity(rot, dx, dy, 7)

        elif self.step == 4:
            # move the purple ball
            if self.time % 0.5 < 0.03:
                if self.virtual_leader_pos[1] < 5.5:
                    self.virtual_leader_pos[0] += 0.01
                    self.virtual_leader_pos[1] += 0.03
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 1)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)

            # calculate the impact of virtual leader
            dif_leader_x = self.virtual_leader_pos[0] - pos[0]
            dif_leader_y = self.virtual_leader_pos[1] - pos[1]
            leader_distance = math.sqrt(dif_leader_x * dif_leader_x + dif_leader_y * dif_leader_y)
            sum_gradient = self.compute_potential_field_gradient(leader_distance, 5, 0.35, 5)
            vel_norm = np.linalg.norm([dif_leader_x, dif_leader_y])  # norm of
            if vel_norm < 0.01:
                vel_norm = 0.01
            vir_theta = np.arctan2(dif_leader_y / vel_norm, dif_leader_x / vel_norm)
            dx += sum_gradient * np.cos(vir_theta)
            dy += sum_gradient * np.sin(vir_theta)
            self.compute_and_set_velocity(rot, dx, dy, 7)

        elif self.step == 5:
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 1)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)


            # calculate the impact of virtual leader
            dif_leader_x = self.virtual_leader_pos[0] - pos[0]
            dif_leader_y = self.virtual_leader_pos[1] - pos[1]
            leader_distance = math.sqrt(dif_leader_x * dif_leader_x + dif_leader_y * dif_leader_y)
            sum_gradient = self.compute_potential_field_gradient(leader_distance, 5, 0.6, 5)
            vel_norm = np.linalg.norm([dif_leader_x, dif_leader_y])  # norm of
            if vel_norm < 0.01:
                vel_norm = 0.01
            vir_theta = np.arctan2(dif_leader_y / vel_norm, dif_leader_x / vel_norm)
            dx += sum_gradient * np.cos(vir_theta)
            dy += sum_gradient * np.sin(vir_theta)
            self.compute_and_set_velocity(rot, dx, dy, 7)

        elif self.step == 6:
            # move to the place under the red ball
            if self.id == 4:
                dx = 0
                dy = 0
                if pos[0] < 4.8:
                    dx = 1
                    dy = 0
                elif pos[1] > 0.5:
                    dx = 0
                    dy = -1
                self.compute_and_set_velocity(rot, dx, dy, 7)
            else:
                dx = 0.
                dy = 0.
                if messages:
                    for m in messages:
                        dif_x = m[1]["pos"][0] - pos[0]
                        dif_y = m[1]["pos"][1] - pos[1]
                        if m[0] == self.step6_topo[self.id]:
                            dx += dif_x
                            dy += dif_y
                        distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                        repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.2, 0.3)
                        vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                        if vel_norm < 0.01:
                            vel_norm = 0.01
                        rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                        dx += repulsive_gradient * np.cos(rep_theta)
                        dy += repulsive_gradient * np.sin(rep_theta)

                self.compute_and_set_velocity(rot, dx, dy, 7.5)

        elif self.step == 7:
            # circle around the red ball with radius = 0.5
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 1)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)

            # calculate the impact of virtual leader
            dif_leader_x = self.virtual_leader2_pos[0] - pos[0]
            dif_leader_y = self.virtual_leader2_pos[1] - pos[1]
            leader_distance = math.sqrt(dif_leader_x * dif_leader_x + dif_leader_y * dif_leader_y)
            sum_gradient = self.compute_potential_field_gradient(leader_distance, 5, 0.5, 5)
            vel_norm = np.linalg.norm([dif_leader_x, dif_leader_y])  # norm of
            if vel_norm < 0.01:
                vel_norm = 0.01
            vir_theta = np.arctan2(dif_leader_y / vel_norm, dif_leader_x / vel_norm)
            dx += sum_gradient * np.cos(vir_theta)
            dy += sum_gradient * np.sin(vir_theta)
            self.compute_and_set_velocity(rot, dx, dy, 7)

        elif self.step == 8:
            # enclose to move the red ball to the red square
            # circle around the red ball with radius = 0.35
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 1)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)

            # calculate the impact of virtual leader
            dif_leader_x = self.virtual_leader2_pos[0] - pos[0]
            dif_leader_y = self.virtual_leader2_pos[1] - pos[1]
            leader_distance = math.sqrt(dif_leader_x * dif_leader_x + dif_leader_y * dif_leader_y)
            sum_gradient = self.compute_potential_field_gradient(leader_distance, 5, 0.35, 5)
            vel_norm = np.linalg.norm([dif_leader_x, dif_leader_y])  # norm of
            if vel_norm < 0.01:
                vel_norm = 0.01
            vir_theta = np.arctan2(dif_leader_y / vel_norm, dif_leader_x / vel_norm)
            dx += sum_gradient * np.cos(vir_theta)
            dy += sum_gradient * np.sin(vir_theta)
            self.compute_and_set_velocity(rot, dx, dy, 7)

        elif self.step == 9:
            # move the red ball to red square
            if self.time % 0.5 < 0.03:
                if self.virtual_leader2_pos[1] < 5.5:
                    self.virtual_leader2_pos[0] += (-0.025)
                    self.virtual_leader2_pos[1] += 0.025
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 1)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)

            # calculate the impact of virtual leader
            dif_leader_x = self.virtual_leader2_pos[0] - pos[0]
            dif_leader_y = self.virtual_leader2_pos[1] - pos[1]
            leader_distance = math.sqrt(dif_leader_x * dif_leader_x + dif_leader_y * dif_leader_y)
            sum_gradient = self.compute_potential_field_gradient(leader_distance, 5, 0.35, 5)
            vel_norm = np.linalg.norm([dif_leader_x, dif_leader_y])  # norm of
            if vel_norm < 0.01:
                vel_norm = 0.01
            vir_theta = np.arctan2(dif_leader_y / vel_norm, dif_leader_x / vel_norm)
            dx += sum_gradient * np.cos(vir_theta)
            dy += sum_gradient * np.sin(vir_theta)
            self.compute_and_set_velocity(rot, dx, dy, 7)

            pass
        elif self.step == 10:
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 1)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)

            # calculate the impact of virtual leader
            dif_leader_x = self.virtual_leader2_pos[0] - pos[0]
            dif_leader_y = self.virtual_leader2_pos[1] - pos[1]
            leader_distance = math.sqrt(dif_leader_x * dif_leader_x + dif_leader_y * dif_leader_y)
            sum_gradient = self.compute_potential_field_gradient(leader_distance, 5, 0.6, 5)
            vel_norm = np.linalg.norm([dif_leader_x, dif_leader_y])  # norm of
            if vel_norm < 0.01:
                vel_norm = 0.01
            vir_theta = np.arctan2(dif_leader_y / vel_norm, dif_leader_x / vel_norm)
            dx += sum_gradient * np.cos(vir_theta)
            dy += sum_gradient * np.sin(vir_theta)
            self.compute_and_set_velocity(rot, dx, dy, 7)

        elif self.step == 11:
            # move the swarm into the room
            if self.id == 0:
                dx = 0
                dy = 0
                if pos[0] < 2.4:
                    dx = 1
                    dy = 0
                elif pos[1] > (-1.5):
                    dx = 0
                    dy = -1
                self.compute_and_set_velocity(rot, dx, dy, 7)
            else:
                dx = 0.
                dy = 0.
                if messages:
                    for m in messages:
                        dif_x = m[1]["pos"][0] - pos[0]
                        dif_y = m[1]["pos"][1] - pos[1]
                        if m[0] == self.step11_topo[self.id]:
                            dx += dif_x
                            dy += dif_y
                        distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                        repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.5, 0.5)
                        vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                        if vel_norm < 0.01:
                            vel_norm = 0.01
                        rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                        dx += repulsive_gradient * np.cos(rep_theta)
                        dy += repulsive_gradient * np.sin(rep_theta)

                self.compute_and_set_velocity(rot, dx, dy, 7.5)

        elif self.step == 12:
            dx = -1
            dy = 0
            self.compute_and_set_velocity(rot, dx, dy, 7)

        elif self.step == 13:
            dx = 0.
            dy = 0.
            if messages:
                for m in messages:
                    dif_x = m[1]["pos"][0] - pos[0]
                    dif_y = m[1]["pos"][1] - pos[1]
                    dx += dif_x - (self.step13_ref[m[0]][0] - self.step13_ref[self.id][0])
                    dy += dif_y - (self.step13_ref[m[0]][1] - self.step13_ref[self.id][1])

                    distance = math.sqrt(dif_x * dif_x + dif_y * dif_y)
                    repulsive_gradient = self.compute_repulsive_force_gradient(distance, 0.4, 0.5)
                    vel_norm = np.linalg.norm([dif_x, dif_y])  # norm
                    if vel_norm < 0.01:
                        vel_norm = 0.01
                    rep_theta = np.arctan2(dif_y / vel_norm, dif_x / vel_norm)
                    dx += repulsive_gradient * np.cos(rep_theta)
                    dy += repulsive_gradient * np.sin(rep_theta)

                self.compute_and_set_velocity(rot, dx, dy, 7)
        elif self.step == 14:
            self.set_wheel_velocity([0, 0])

        # check whether should goto next step every second
        if self.time % 1 < 0.03:
            print("CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            if self.step == 0:
                print(abs(self.step1_ref[self.id][1] - pos[1]))
                if 0.4 < abs(self.step1_ref[self.id][1] - pos[1]) < 0.55 and self.time > 9:
                    self.next_step = 1
            elif self.step == 1:
                print("step-1, robot-{}, next-step: {}".format(self.id, self.next_step))
                if self.id == 0:
                    if pos[1] > 2.4:
                        self.next_step = 2
                else:
                    if messages:
                        for m in messages:
                            if m[1]["next_step"] == 2:
                                self.next_step = 2
            elif self.step == 2:
                print("leader distance: {}".format(leader_distance))
                if abs(leader_distance - 0.5) < 0.1:
                    self.next_step = 3
            elif self.step == 3:
                print("Step 3 - leader distance: {}".format(leader_distance))
                if abs(leader_distance - 0.35) < 0.1:
                    self.next_step = 4
            elif self.step == 4:
                print("Step 4 - leader distance: {}".format(leader_distance))
                if abs(leader_distance - 0.35) < 0.1 and self.virtual_leader_pos[1] > 5.5:
                    self.next_step = 5
            elif self.step == 5:
                print("Step 5 - leader distance: {}".format(leader_distance))
                if abs(leader_distance - 0.6) < 0.1:
                    self.next_step = 6
            elif self.step == 6:
                print("Step 6")
                if self.id == 1:
                    if pos[1] < 3.9:
                        self.next_step = 7
                else:
                    if messages:
                        for m in messages:
                            if m[1]["next_step"] == 7:
                                self.next_step = 7
            elif self.step == 7:
                print("Step 7 - leader distance: {}".format(leader_distance))
                if abs(leader_distance - 0.5) < 0.1:
                    self.next_step = 8
            elif self.step == 8:
                print("Step 8 - leader distance: {}".format(leader_distance))
                if abs(leader_distance - 0.35) < 0.1:
                    self.next_step = 9
            elif self.step == 9:
                print("Step 9 - leader distance: {}".format(leader_distance))
                if abs(leader_distance - 0.35) < 0.1 and self.virtual_leader2_pos[1] > 5.5:
                    self.next_step = 10
            elif self.step == 10:
                print("Step 10 - leader distance: {}".format(leader_distance))
                if abs(leader_distance - 0.6) < 0.1:
                    self.next_step = 11
            elif self.step == 11:
                print("Step 11 - moving into the room")
                if self.id == 5:
                    if pos[1] < 1.5:
                        self.next_step = 12
                else:
                    if messages:
                        for m in messages:
                            if m[1]["next_step"] == 12:
                                self.next_step = 12
            elif self.step == 12:
                print("Step 12")
                if pos[0] < 1.8:
                    self.next_step = 13
            elif self.step == 13:
                print("Step 13", abs(self.step1_ref[self.id][1] - pos[1]))
            elif self.step == 14:
                print("Step 14")

        # check step status of neighbors to synchronize and update
        convert_flag = True
        if messages:
            for m in messages:
                if m[1]["next_step"] != self.next_step:
                    convert_flag = False
        if convert_flag:
            self.step = self.next_step

        # print(self.id, self.next_step)
        self.time += self.dt
