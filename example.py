from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os, time, requests, cv2, glob, torch, shutil
from tqdm.auto import tqdm
from random import randint

#Load the file contaning list of classes on which our model is trained on
def load_classes(path_to_classes_file):
    with open(path_to_classes_file, "r") as file:
        data = file.readlines()
    recaptcha_classes = list()
    for d in data:
        recaptcha_classes.append(d.strip("\n"))
    return recaptcha_classes


#Now load the model, here we are loading the yolov5 model
def load_model(path_to_weights, path_to_classes_file="./classes.txt"):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights), load_classes(path_to_classes_file)

#This function coverts the bounding box cordinates returned by yolo into format that we use in array format of images
def convert(xmin, ymin, xmax, ymax):
    x1 = int(xmin)
    x2 = int(xmax)
    y1 = int(ymin)
    y2 = int(ymax)
    return [y1,x1,y2,x2]

#These directories will be used to temporary store images that will be downloaded from recaptcha
if not os.path.exists("./images"):
    os.mkdir("./images")
if not os.path.exists("./cropped_images"):
    os.mkdir("./cropped_images")

#Calling the loading model function and passing it the path to our custom weights for yolo5
model,recap_classes = load_model('./yolov5/runs/train/yolov5x_recaptcha/weights/best.pt')


#Now using selenium we intialize our driver with certain options
options = webdriver.ChromeOptions()
options.add_argument("--log-level=3")
options.binary_location = "C:\\Program Files\\Google\\Chrome Beta\\Application\\chrome.exe" #You need to change the path where your chrome binary is. This is usally the default location.
options.add_argument("user-agent=User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.29 Safari/537.36")
bot = webdriver.Chrome(service=Service("chromedriver.exe"), options=options)
bot.get("https://www.google.com/recaptcha/api2/demo")
wait = WebDriverWait(bot, 10)
bot.find_element(By.ID, "recaptcha-demo").click()
wait.until(EC.frame_to_be_available_and_switch_to_it(bot.find_element(By.XPATH, "//iframe[@title='recaptcha challenge expires in two minutes']")))
time.sleep(3)


while(1):
    #First capture the recaptcha class for which we need to solve the recaptcha....example: select images with cars/bus etc
    try:
        captcha_class = bot.find_element(By.XPATH, "//div[@id='rc-imageselect']/div[2]/div[1]/div[1]/div[1]/strong").text
    except:
        captcha_class = bot.find_element(By.XPATH, "//div[@id='rc-imageselect']/div[2]/div[1]/div[1]/div[2]/strong").text
    
    # Then we make sure that the class for which we have to solve is one of the classes on which our model was trained
    # Otherwise we just ignore this run and refresh the recaptcha

    for c in recap_classes:
        continue_to_next_iteration = False
        if c.find(captcha_class) == -1:
            continue_to_next_iteration = True
        else:
            continue_to_next_iteration = False
    if not continue_to_next_iteration:
        bot.quit()
        time.sleep()
        continue
    
    #Now we download all the images that are shown in the recaptcha to our ./images directory for temporary use
    total_images = 0
    rows = bot.find_elements(By.XPATH, "//div[@id='rc-imageselect-target']/table/tbody/tr")
    for row in range(len(rows)):
        cols = bot.find_elements(By.XPATH, f"//div[@id='rc-imageselect-target']/table/tbody/tr[{row+1}]/td")
        for col in range(len(cols)):
            img = bot.find_element(By.XPATH, f"//div[@id='rc-imageselect-target']/table/tbody/tr[{row+1}]/td[{col+1}]/div/div[1]/img").get_attribute("src")
            img_data = requests.get(img).content
            with open(f"images/image{total_images}.jpg", 'wb') as handler:
                handler.write(img_data)
            total_images+=1
    
    #The thing is that images in recaptcha even tho appear as seperate images in backend its one image split into multiple images
    #So above when we download the images, we actually get all images shown in recaptcha in one image instead of seperate images
    #So we need to manually split those images to create sepereate images as shown in recaptcah
    #There are two types of pattern....One where we have 9 images, or the one where have 16 images, so we split differently for each
    if total_images == 9:
        divider = 3
    elif total_images == 16:
        divider = 4


    #Sometime when we click on recaptcha a new image appears at its place, in these scenarios only that single image
    #got updated, so when we solved one run and now to have run from start again to solve the new images, so we have make
    #sure we only solve for the new images that appeared instead of running all the images through the model again, which
    #is basically waste of resources(especially if you are hosting the model on some cloud sevcie)
    full_dim_image_found = False
    img_numbers = list()
    for i in range(total_images):
        img = cv2.imread(f'./images/image{i}.jpg')
        if divider == 3:
            if not (img.shape[0] == 300 and img.shape[1] == 300):
                img_numbers.append(i)
        else:
            if not (img.shape[0] == 450 and img.shape[1] == 450):
                img_numbers.append(i)
    
    #If img_numbers list is empty that means its our first run all images are stored in one single image, we need to consider all the images
    #and then we split that single image into 9 seperate images as shown in recaptcha to solve them seperately
    #if there are 16 images then there is no need to split images as the recaptcha with 16 images have to be solved together since we have
    #select an object that spans over multiple images, so if we divide those images we might not be able to detect the object
    #Else if img_numbers is not empty then it just means we only have to solve the new images that appeared and we just copy those new images
    #to our cropped images directory

    if len(img_numbers) < 1:
        if divider == 3:
            img_numbers = range(9)
        x_factor, y_factor = img.shape[0]//divider , img.shape[1]//divider
        img_save = 0
        cropped_images_cord = list()
        for x in range(len(cols)):
            for y in range(len(rows)):
                cropped_image = img[x_factor*x:x_factor*(x+1), y_factor*y:y_factor*(y+1)]
                cropped_images_cord.append([x_factor*x, y_factor*y, x_factor*(x+1), y_factor*(y+1)])
                cv2.imwrite(f"./cropped_images/image{img_save}.jpg", cropped_image)
                img_save+=1
        else:
            img_numbers = range(16)
    else:
        for i in img_numbers:
            shutil.copy(f"./images/image{i}.jpg", f"./cropped_images/image{i}.jpg")


    #In this part we load the images and pass them to the model for inference
    #We need to solve differently for each pattern(9 or 16 images)
    images = os.listdir("./cropped_images/")
    class_list = list()
    object__cord_list = list()
    no_object = False
    if divider ==3:
        for image in tqdm(images, desc="Predicting Results "):
            result = model("./cropped_images/"+image)
            objects_in_image = result.pandas().xyxy[0]['name'].tolist()
            class_list.append(objects_in_image)
    else:
        for i in tqdm(range(1), desc="Predicting Results "):
            result = model(f"./images/image{total_images-1}.jpg")
            result.save()
        temp_df = result.pandas().xyxy[0]['name'].tolist()
        temp_id = [id for id in range(len(temp_df)) if temp_df[id].find(captcha_class) != -1]
        if len(temp_id) > 0:
            objects_cord = result.pandas().xyxy[0].loc[temp_id]
            objects_cord =  objects_cord[['xmin','ymin','xmax','ymax']].values.tolist()
            object__cord_list = [convert(obj[0], obj[1], obj[2], obj[3]) for obj in objects_cord]
        else:
            no_object = True
    
    #So, if we have 9 images, which means each row contain 3 images/col and each col in row contain 3 images
    #We need to go over each image one by one and see what did the model predict for that image, so the class for which we are 
    #trying to solve is predicted in that image we click it otherwise we move to the next image
    #But if we have 16 images, the process is solve is pretty different, we do like before go over one picture in row one by one
    #but instead of just checking if the model predicted the class we are solving for in the image we now instead try to find
    #the bounding box cordinate of the object we predicted intersect with this image, if they do we select it otherwise we move on
    #to the next image
    if divider == 3:  
        image_count = 0
        count=0
        rows = bot.find_elements(By.XPATH, "//div[@id='rc-imageselect-target']/table/tbody/tr")
        for row in range(len(rows)):
            cols = bot.find_elements(By.XPATH, f"//div[@id='rc-imageselect-target']/table/tbody/tr[{row+1}]/td")
            for col in range(len(cols)):
                img = bot.find_element(By.XPATH, f"//div[@id='rc-imageselect-target']/table/tbody/tr[{row+1}]/td[{col+1}]")
                if image_count in img_numbers:
                    for i in range(len(class_list[count])):
                        if class_list[count][i].find(captcha_class) != -1:
                            img.click()
                            time.sleep(4)
                            break
                    count+=1
                image_count+=1
    else:
        if not no_object:
            print(object__cord_list)
            print(cropped_images_cord)
            image_count = 0
            rows = bot.find_elements(By.XPATH, "//div[@id='rc-imageselect-target']/table/tbody/tr")
            for row in range(len(rows)):
                cols = bot.find_elements(By.XPATH, f"//div[@id='rc-imageselect-target']/table/tbody/tr[{row+1}]/td")
                for col in range(len(cols)):
                    img = bot.find_element(By.XPATH, f"//div[@id='rc-imageselect-target']/table/tbody/tr[{row+1}]/td[{col+1}]")
                    for j in range(len(object__cord_list)):
                        if ((object__cord_list[j][0] in range(cropped_images_cord[image_count][0], cropped_images_cord[image_count][2]) and object__cord_list[j][1] in range(cropped_images_cord[image_count][1], cropped_images_cord[image_count][3])) or
                                (object__cord_list[j][0] in range(cropped_images_cord[image_count][0], cropped_images_cord[image_count][2]) and object__cord_list[j][3] in range(cropped_images_cord[image_count][1], cropped_images_cord[image_count][3])) or
                                (object__cord_list[j][2] in range(cropped_images_cord[image_count][0], cropped_images_cord[image_count][2]) and object__cord_list[j][1] in range(cropped_images_cord[image_count][1], cropped_images_cord[image_count][3])) or
                                (object__cord_list[j][2] in range(cropped_images_cord[image_count][0], cropped_images_cord[image_count][2]) and object__cord_list[j][3] in range(cropped_images_cord[image_count][1], cropped_images_cord[image_count][3])) or
                                (object__cord_list[j][0] in range(cropped_images_cord[image_count][0], cropped_images_cord[image_count][2]) and randint(cropped_images_cord[image_count][1], cropped_images_cord[image_count][3]) in range(object__cord_list[j][1], object__cord_list[j][3])) or
                                (object__cord_list[j][2] in range(cropped_images_cord[image_count][0], cropped_images_cord[image_count][2]) and randint(cropped_images_cord[image_count][1], cropped_images_cord[image_count][3]) in range(object__cord_list[j][1], object__cord_list[j][3])) or
                                (randint(cropped_images_cord[image_count][0], cropped_images_cord[image_count][2]) in range(object__cord_list[j][0], object__cord_list[j][2]) and object__cord_list[j][1] in range(cropped_images_cord[image_count][1], cropped_images_cord[image_count][3])) or
                                (randint(cropped_images_cord[image_count][0], cropped_images_cord[image_count][2]) in range(object__cord_list[j][0], object__cord_list[j][2]) and object__cord_list[j][3] in range(cropped_images_cord[image_count][1], cropped_images_cord[image_count][3]))):
                            img.click()
                            time.sleep(4)
                            break
                    image_count+=1

    #Now we remove all the images we stored tempo
    files = glob.glob('./cropped_images/*')
    for f in files:
        os.remove(f)
    files = glob.glob('./images/*')
    for f in files:
        os.remove(f)

    #Now we check if the recaptcha has been solved or if some new images appeared that we need to solve for
    cont = True
    try:
        bot.find_element(By.XPATH, "//button[@id='recaptcha-verify-button']").click()
    except:
        try:
            bot.find_element(By.XPATH, "//*[contains(text(), 'Skip')]").click()
        except:
            try:
                bot.find_element(By.XPATH,"//*[contains(text(), 'Please select all matching images.')]")
            except:
                try:
                    bot.find_element(By.XPATH, "//*[contains(text(), 'Please also check the new images.')]")
                except:
                    try:
                        bot.find_element(By.XPATH,"//*[contains(text(), 'Please try again')]")
                    except:
                        cont = False
    if not cont:
        break

    time.sleep(5)

print("Solved ReCaptcha!!!")
time.sleep(10)
bot.quit()




