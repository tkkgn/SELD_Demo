
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd
import os
import storage_params as sp



class Animation():
    
    #Enter your own video name .mp4
    VIDEO_PATH = os.path.join(sp.dir_path,sp.anim_file)
    
    def __init__(self,data, name):
        #initializing data
        self.data = data
        self.name = name
        self.stream = self.data_stream()
        
        
        # initializing a figure in 
        # which the graph will be plotted
        self.fig,self.ax = plt.subplots(figsize=(10,10)) 



        #setup video params
        self.cur_frame =0
        self.max_frames = 300
        self.temp = 0
        self.current=0
        
        #self.title = self.ax.set_title("the frame number: {}".format(self.cur_frame))
        #self.colorbar = plt.colorbar()
        
        #setup FuncAnimation
        self.anni = animation.FuncAnimation(self.fig, 
                                            self.update, 
                                            frames = self.max_frames, 
                                            interval = 100, 
                                            init_func= self.setup_plot, 
                                            blit = True)
        
        #save animation to file
        writervideo = animation.FFMpegWriter(fps=10)
        self.anni.save(self.VIDEO_PATH,writervideo)
        

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        # marking the x-axis and y-axis
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.grid()
        
        #x, y, name, = next(self.stream)
        self.ax.set_xlim([-0.5,0.5])
        self.ax.set_ylim([-0.5,0.5])
        self.ax.scatter(0,0,c='r')
        self.ax.annotate("Mic",(0,0))
        self.scat = self.ax.scatter([], [], s=100)
        self.__annotations__= self.ax.annotate("",(0,0))
        self.__annotations1__= self.ax.annotate("",(0,0))
        self.__annotations2__= self.ax.annotate("",(0,0))
        self.__annotations3__= self.ax.annotate("",(0,0))
        
        
        # For FuncAnimation's sake, we need to return the artist we'll be using
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,self.__annotations__,self.__annotations1__, self.__annotations2__,self.__annotations3__
    

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        
        #s, c = np.random.random((self.numpoints, 2)).T
        while self.temp < len(self.name):
            if self.cur_frame == int(self.data[self.temp,0]):
                #print(self.temp)
                self.temp += 1
                continue
            xtemp=self.data[self.current:self.temp,1]
            ytemp=self.data[self.current:self.temp,2]
            nameT = self.name[self.current:self.temp]
            print(self.cur_frame,xtemp,ytemp,nameT)
            self.current=self.temp
            self.cur_frame += 1

            yield xtemp,ytemp,nameT

        #frame 299
        xtemp=self.data[self.current:self.temp,1]
        ytemp=self.data[self.current:self.temp,2]
        nameT = self.name[self.current:self.temp]
        print(self.cur_frame,xtemp,ytemp,nameT)
        self.current=self.temp
        self.cur_frame += 1
        yield xtemp,ytemp,nameT
    



    
    def update(self,f):
        """Update the scatter plot."""
        x,y,name = next(self.stream)
        self.ax.clear()
        #self.title.set_text("the frame number: {}".format(f))
        self.setup_plot()
        
        
        #print(x,y)
        # Set x and y data...
        self.scat.set_offsets(np.column_stack((x,y)))
        
        if len(name) == 1:
            self.__annotations__= self.ax.annotate(name[0],(x[0],y[0]))
            return self.scat,self.__annotations__
        
        if len(name) == 2:
            self.__annotations__= self.ax.annotate(name[0],(x[0],y[0]))
            self.__annotations1__= self.ax.annotate(name[1],(x[1],y[1]))
            return self.scat, self.__annotations__, self.__annotations1__
        
        if len(name) == 3:
            self.__annotations__= self.ax.annotate(name[0],(x[0],y[0]))
            self.__annotations1__= self.ax.annotate(name[1],(x[1],y[1]))
            self.__annotations2__= self.ax.annotate(name[2],(x[2],y[2]))
            return self.scat, self.__annotations__, self.__annotations1__, self.__annotations2__
        
        if len(name) == 4:
            self.__annotations__= self.ax.annotate(name[0],(x[0],y[0]))
            self.__annotations1__= self.ax.annotate(name[1],(x[1],y[1]))
            self.__annotations2__= self.ax.annotate(name[2],(x[2],y[2]))
            self.__annotations3__= self.ax.annotate(name[3],(x[3],y[3]))
            return self.scat,self.__annotations__,self.__annotations1__, self.__annotations2__,self.__annotations3__
        
        
        #self.annotation.new_frame_seq(annotation)


        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,
    


    

#test fi
if __name__ == '__main__':
    df = pd.read_csv("test.csv")
    data = np.column_stack((df["Frames"].to_numpy(),df["X"].to_numpy(),df["Y"].to_numpy()))
    #st.write(data)
    name = df["Class Name"].to_numpy()
    anme = Animation(data,name)
     
    #plt.show()

    