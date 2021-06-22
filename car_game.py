import pyautogui as gui
import time
from getting_state import Frame
from get_clip import VideoFromImages

class Car():
    def __init__(self):
        
        self.actionSpace = [0, 1, 2, 3, 4, 5] ## up, down, left, right, back right, back left
        self.previousScore = 0  ## previous score
        self.previousCollision = 0
        self.startTime = time.time()
        self.epDuration = 300    ## episode duration in seconds
        self.maxScore = 50       ## max score in game
        self.screenPosX = 1855   ## for reset env
        self.screenPosY = 903    ## for reset env
        self.negReward = -0.1
        self.posReward = 100 
        self.frame = Frame()
    
    def get_score(self):
        try:
            time.sleep(.1)
            filePath = "/run/user/1000/gvfs/mtp:host=SAMSUNG_SAMSUNG_Android_988dd1424c494c385a30/Phone/Android/data/com.soothscier.DrivingMatter/files"
            fileName = filePath + "/DrivingScore.txt"
            with open(fileName, 'r') as f:
                return int(f.readline())
        except FileNotFoundError:
            print("\n\t--> Score file is not found")
            exit()

    def get_collision_info(self):
        """ 
            getting information about collision triggered event in AR game

        """
        try:
            time.sleep(.1)
            filePath = "/run/user/1000/gvfs/mtp:host=SAMSUNG_SAMSUNG_Android_988dd1424c494c385a30/Phone/Android/data/com.soothscier.DrivingMatter/files"
            fileName = filePath + "/BoundaryCollision.txt"
            with open(fileName, 'r') as f:
                return int(f.read())

        except FileNotFoundError: 
            print("\n\t--> Boundary file is not found")
            exit()
        except ValueError:
            pass

    def perform_action(self, action):
        """
            Performing an action in AR game using laptop.
        """
        if action == 0: ## forward
            gui.press('s')
        
        elif action == 1: ## reverse
            gui.press('w')

        elif action == 2: ## turn left
            gui.keyDown('a')
            gui.press('w')
            gui.press('w')
            gui.keyUp('a')
            

        elif action == 3: ## turn right
            gui.keyDown('d')
            gui.press('w')
            gui.press('w')
            gui.keyUp('d')

        elif action == 4: ## reverse left
            gui.keyDown('a')
            gui.press('s')
            gui.press('s')
            gui.keyUp('a')

        elif action == 5: ## reverse right
            gui.keyDown('d')
            gui.press('s')
            gui.press('s')
            gui.keyUp('d')

    def timer(self):
        """
         time threshold to get all diamonds/portions
        
        """
        end_time = time.time()
        if (end_time - self.startTime) >= self.epDuration:
            self.startTime = time.time()
            return True 
        return False

        
    def step(self, action):
        """
            getting reward after performing an action.
        
        """
        self.perform_action(action)
        done = False
        collision = self.get_collision_info()
        print(collision)
        score = self.get_score()
        state = self.frame.get_frame()
        reward = 0

        if self.previousCollision < collision:
            ## wall collision detection
            reward += self.negReward * 5

        if self.previousScore == score:
            ## same score in game
            reward += self.negReward

        if score > self.previousScore:
            ## picked portion/diamond
            reward += (score - self.previousScore) * 10

        if self.timer() and score < self.maxScore:
            ## time's up and have not collected all portions/diamonds
            reward += self.negReward * 5
            done = True

        if score >= self.maxScore :
            ## all portions/diamonds are collected
            print("\n\tHurray..!! Finished\n")
            reward += self.posReward
            done = True

        self.previousScore = score
        self.previousCollision = collision
        return state, reward, done, {}
   
    def reset(self):
        """
            Resetting environment for next episode.
            
        """
        ## place all portions/diamonds
        self.previousScore = 0
        self.previousCollision = 0
        state = self.frame.get_frame()
        gui.click(self.screenPosX, self.screenPosY)  ## clicking 'environment' button in game
        return state

    def render(self):
        ## merged all images/frames and make a video clip
        ## computationally costly at my laptop for now
        
        ins = VideoFromImages()
        ins.generate_video()
        
        