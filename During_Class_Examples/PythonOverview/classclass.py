# Create a class for a class with a list of students, and the name of the class
class Class:
    def __init__(self, name, students):
        self.name = name
        self.students = students

    def add_student(self, student):
        self.students.append(student)

    def get_students(self):
        return self.students

# Create a class for a student
class Student:
    def __init__(self, name, age, hobbies, luckynumbe):
        self.name = name
        try: 
            self.age = age/luckynumbe
            # Throw a different exception
            raise ValueError("Very bad text here")
        except ZeroDivisionError:
            print("You have a very unlucky number")
            self.age = 50
        except Exception as e:
            print("It cached our value error")
            print(e)
        self.hobbies = hobbies

    def get_name(self):
        return self.name
    
    def set_name(self, name):
        self.name = name
    
        