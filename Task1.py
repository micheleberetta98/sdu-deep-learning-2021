import random, os, shutil
import sys

#create seed
# seed_value = random.randrange(sys.maxsize)
#
# #save the seed
# print('Seed value:',  seed_value)
#
# #Seed the random number generator
# random.seed(seed_value)
# num = (random.randrange(1, 1100))
#
# print("random number:", num)

source ="D:\\Project_SDU\\data"
test_dest ="C:\\Users\\qnarari\\Documents\\GitHub\\sdu-deep-learning-2021\\final-project\\testing"
val_dest ="C:\\Users\\qnarari\\Documents\\GitHub\\sdu-deep-learning-2021\\final-project\\validation"
train_dest="C:\\Users\\qnarari\\Documents\\GitHub\\sdu-deep-learning-2021\\final-project\\training"

def copy_random (no_of_files,status, dest):
    for i in range(no_of_files):
        random_file = random.choice([x for x in os.listdir(source) if status in x])
        print("%d} %s"%(i+1,random_file))
        source_file = "%s\%s" % (source, random_file)
        print(source)
        dest_file=dest+"\\"+status
        print(dest_file)
        shutil.move(source_file, dest_file)

training_files=int(input("Enter The Number of Files To Select For Training: "))
val_files=int(input("Enter The Number of Files To Select For Validation: "))
test_files=int(input("Enter The Number of Files To Select For Testing: "))

copy_random(training_files,"normal",test_dest)
print("\n\n" + "$" * 33 + "[ Files Moved Successfully ]" + "$" * 33)
copy_random(training_files,"pneumonia",test_dest)
print("\n\n" + "$" * 33 + "[ Files Moved Successfully ]" + "$" * 33)

copy_random(val_files,"normal",val_dest)
print("\n\n" + "$" * 33 + "[ Files Moved Successfully ]" + "$" * 33)
copy_random(val_files,"pneumonia",val_dest)
print("\n\n" + "$" * 33 + "[ Files Moved Successfully ]" + "$" * 33)

copy_random(test_files,"normal",train_dest)
print("\n\n" + "$" * 33 + "[ Files Moved Successfully ]" + "$" * 33)
copy_random(test_files,"pneumonia",train_dest)
print("\n\n" + "$" * 33 + "[ Files Moved Successfully ]" + "$" * 33)

