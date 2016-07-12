class GaitData:
	'This class contain all the information of a gait'
	def __init__(self):
		self.joint_descriptors = ['Head', 'Shoulder-Center', 'Shoulder-Right', 'Shoulder-Left', 'Elbow-Right', 'Elbow-Left', 'Wrist-Right', 'Wrist-Left','Hand-Right', 'Hand-Left', 'Spine', 'Hip-centro', 'Hip-Right', 'Hip-Left', 'Knee-Right', 'Knee-Left','Ankle-Right', 'Ankle-Left', 'Foot-Right', 'Foot-Left']
		
		self.head = 0
		self.shoulderCenter = 1
		self.shoulderRight = 2
		self.shoulderLeft = 3
		self.elbowRight = 4
		self.elbowLeft = 5
		self.wristRight = 6
		self.wristLeft = 7
		self.handRight = 8
		self.handLeft = 9
		self.spine = 10
		self.hipCenter = 11
		self.hipRight = 12
		self.hipLeft = 13
		self.kneeRight = 14
		self.kneeLeft = 15
		self.ankleRight = 16
		self.ankleLeft = 17
		self.footRight = 18
		self.footLeft = 19

		self.points = []
		self.id = 1

	def setId(self,id):
		self.id = id
	def getId(self):
		return self.id

