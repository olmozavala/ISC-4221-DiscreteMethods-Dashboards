# Instantiate two students and one class
from classclass import Student, Class

student1 = Student("John", 20, ["reading", "writing"], 1)
student2 = Student("Jane", 21, ["reading", "writing"], 0)
class1 = Class("Class 1", [student1, student2])

# Add a student to the class
class1.add_student(Student("Jim", 22, ["reading", "writing"], 3))

# Get the students in the class
print(class1.get_students())

student1.set_name("Zeus")
print(student1.get_name())