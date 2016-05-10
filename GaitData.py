class GaitData:
	'This class contain all the information of a gait'
	def __init__(self):
		self.joint_descriptors = ['Head', 'Shoulder-Center', 'Shoulder-Right', 'Shoulder-Left', 'Elbow-Right', 'Elbow-Left', 'Wrist-Right', 'Wrist-Left','Hand-Right', 'Hand-Left', 'Spine', 'Hip-centro', 'Hip-Right', 'Hip-Left', 'Knee-Right', 'Knee-Left','Ankle-Right', 'Ankle-Left', 'Foot-Right', 'Foot-Left']
		
		self.head = 0
		self.shoulder_center = 1
		self.shoulder_right = 2
		self.shoulder_left = 3
		self.elbow_right = 4
		self.elbow_left = 5
		self.wrist_right = 6
		self.wrist_left = 7
		self.hand_right = 8
		self.hand_left = 9
		self.spine = 10
		self.hip_center = 11
		self.hip_right = 12
		self.hip_left = 13
		self.knee_right = 14
		self.knee_left = 15
		self.ankle_right = 16
		self.ankle_left = 17
		self.foot_right = 18
		self.foot_left = 19

		self.points = []


