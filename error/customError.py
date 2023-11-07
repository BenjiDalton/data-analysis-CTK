class customError(Exception):
	def __init__(self, message: str):   
		"""
		Print error message to user.
		"""
		print(message)