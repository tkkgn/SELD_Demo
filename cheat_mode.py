import random 
import pandas as pd
import math



class Coordinate_fake():
    def __init__(self,df):
        self.df = df
        
        #setup fixed coordinates for some class name
        self.fixed_location = {
            'Computer keyboard': 
                [(-1.5,0.5),(-1.5,-0.5),(0.0,-1.0),(1.5,-0.5),(1.5,0.5)],
            'cupboard open/close': (-1.0,1.0),
            'Drawer open/close': (1.0,1.0),
            'Printer': (0.0,1.125),
            'Knock': (-1.25,-1.25)    
        }
        
        self.detect = False
        
    
    def random_inrange(self,x_item,y_item):
        threshold = random.uniform(0.01,0.05)
        x = random.uniform(x_item - threshold, x_item + threshold)
        y= random.uniform(y_item - threshold, y_item + threshold)
        return x,y
    
    def get_closet_distance(self,x_raw,y_raw):
        seed_coors =self.fixed_location.get("Computer keyboard")
        #print(seed_coors)
        #print(type(seed_coors))
        raw_coor =(x_raw,y_raw)
        dest_coor = (0,0)
        min_dist = math.dist(seed_coors[0],raw_coor)
        for i,coor in enumerate(seed_coors):
            temp_dist= math.dist(coor,raw_coor)
            if min_dist > temp_dist:
                min_dist = temp_dist
                dest_coor = coor
            #print(coor)
            #print(temp_dist)
            
        return dest_coor
    
    
    def processing(self):
        temp = (0,0)
        i=0
        for i in range(len(self.df)-1):
            if self.df['Class Name'][i] == "Computer keyboard":
                x,y = self.get_closet_distance(self.df['X'][i],self.df['Y'][i])
                self.df.at[i,'X'],self.df.at[i,'Y'] = self.random_inrange(x,y)
                continue
            
            if self.df['Class Name'][i] in self.fixed_location.keys() and self.detect == False:
                x,y = self.fixed_location.get(self.df['Class Name'][i])
                self.df.at[i,'X'],self.df.at[i,'Y'] = self.random_inrange(x,y)
                temp = self.df.at[i,'X'],self.df.at[i,'Y']
                self.detect = True
                continue
            
            if (self.df['Class Name'][i] == self.df['Class Name'][i+1]) and self.detect:
                x,y = temp
                self.df.at[i,'X'],self.df.at[i,'Y'] = self.random_inrange(x,y)
                temp = self.df.at[i,'X'],self.df.at[i,'Y']
                
                continue
            
            if (self.df['Class Name'][i] != self.df['Class Name'][i+1]) and self.detect:
                x,y= temp
                self.df.at[i,'X'],self.df.at[i,'Y'] = self.random_inrange(x,y)
                self.detect = False
                
                
        #check last index
        if self.df['Class Name'][i] == "Computer keyboard":
            x,y = self.get_closet_distance(self.df['X'][i],self.df['Y'][i])
            self.df.at[i,'X'],self.df.at[i,'Y'] = self.random_inrange(x,y)
            return (self.df)
            
        
        if self.df['Class Name'][i] in self.fixed_location.keys():
            x,y = self.fixed_location.get(self.df['Class Name'][i])
            self.df.at[i,'X'],self.df.at[i,'Y'] = self.random_inrange(x,y)
            #self.detect = True
            
        return self.df    
        
        
                
            
if __name__ == '__main__':
    df = pd.read_csv("test.csv")
    df = Coordinate_fake(df).processing()
    df.to_csv("test2.csv")
    
