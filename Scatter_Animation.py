
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.animation as animation
import pandas as pd
import os
import storage_params as sp
import sys
from timeit import default_timer as timer 
import subprocess

from matplotlib.animation import FFMpegWriter

#Enter your own video name .mp4
VIDEO_PATH = os.path.join(sp.dir_path,sp.anim_file)

class FasterFFMpegWriter(FFMpegWriter):
    '''FFMpeg-pipe writer bypassing figure.savefig.'''
    def __init__(self, **kwargs):
        '''Initialize the Writer object and sets the default frame_format.'''
        super().__init__(**kwargs)
        self.frame_format = 'argb'
        self.canvas_width, self.canvas_height = self.fig.canvas.get_width_height()
        # Open an ffmpeg process
        outf = VIDEO_PATH
        cmdstring = ('ffmpeg', 
                    '-y', '-r', '10', # overwrite, 1fps
                    '-s', '%dx%d' % (self.canvas_width, self.canvas_height), # size of image string
                    '-pix_fmt', 'argb', # format
                    '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                    '-vcodec', 'mpeg4', outf) # output encoding
        self._proc = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)
    

    def grab_frame(self, **savefig_kwargs):
        '''Grab the image information from the figure and save as a movie frame.

        Doesn't use savefig to be faster: savefig_kwargs will be ignored.
        '''
        try:
            # re-adjust the figure size and dpi in case it has been changed by the
            # user.  We must ensure that every frame is the same size or
            # the movie will not save correctly.
            self.fig.set_size_inches(self.canvas_width, self.canvas_height)
            #self.fig.set_dpi(self.dpi)
            # Draw and save the frame as an argb string to the pipe sink
            self.fig.canvas.draw()
            self._proc.stdin.write(self.fig.canvas.tostring_argb()) 
        except (RuntimeError, IOError) as e:
            out, err = self._proc.communicate()
            raise IOError('Error saving animation to file (cause: {0}) '
                      'Stdout: {1} StdError: {2}. It may help to re-run '
                      'with --verbose-debug.'.format(e, out, err)) 


class Animation():
    
    VIDEO_PATH = VIDEO_PATH   
  
    def __init__(self,data, name, argv = False):
        
           
        self.save_video = argv
        
        #initializing data
        self.data = data
        self.name = name
        self.stream = self.data_stream()
        
        # initializing a figure in 
        # which the graph will be plotted
        self.fig,self.ax = plt.subplots(figsize=(9,7)) 
        self.background = plt.imread(sp.background_path)
        
        #fast combining
        self.canvas_width, self.canvas_height = self.fig.canvas.get_width_height()
        
        #set annotation size
        self.text_size=10
        
        #set x,y coordination
        self.x_max = 2
        self.x_min = -2
        self.y_max = 1.5
        self.y_min = -1.5
        
        #setup video params
        self.cur_frame =0
        self.max_frames = 300
        self.temp = 0
        self.current=0
        
        #self.title = self.ax.set_title("the frame number: {}".format(self.cur_frame))
        #self.colorbar = plt.colorbar()
      
        #self.test_using_canvas()
        #setup FuncAnimation

        self.anni = animation.FuncAnimation(self.fig, 
                                            self.update, 
                                            frames = self.max_frames, 
                                            interval = 100, 
                                            init_func= self.setup_plot, 
                                            blit = True)
        if self.save_video:
            #save animation to file
            #writervideo = FasterFFMpegWriter(fps=10,fig = self.fig)
            writervideo = animation.FFMpegWriter(fps=10)
            self.anni.save(VIDEO_PATH,writervideo)
            plt.close()
        else:
            plt.show(block = True)
            plt.close()    
            
    def test_using_canvas(self):
        self.setup_plot()
        
        # Open an ffmpeg process
        outf = VIDEO_PATH
        cmdstring = ('ffmpeg', 
                    '-y', '-r', '10', # overwrite, 1fps
                    '-s', '%dx%d' % (self.canvas_width, self.canvas_height), # size of image string
                    '-pix_fmt', 'argb', # format
                    '-f', 'rawvideo',  '-i', '-', # tell ffmpeg to expect raw video from the pipe
                    '-vcodec', 'mpeg4', outf) # output encoding
        p = subprocess.Popen(cmdstring, stdin=subprocess.PIPE)

        # Draw frames and write to the pipe
        for frame in range(self.max_frames):
            # draw the frame
            self.update(frame)
            self.fig.canvas.draw()

            # extract the image as an ARGB string
            string = self.fig.canvas.tostring_argb()

            # write to pipe
            p.stdin.write(string)

        # Finish up
        p.communicate()
        
        

    def setup_plot(self):
        """Initial drawing of the scatter plot."""
        # marking the x-axis and y-axis
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        
        self.ax.imshow(self.background,extent=[self.x_min,self.x_max,
                                               self.y_min,self.y_max])
        self.ax.grid()
        
        #x, y, name, = next(self.stream)
        self.ax.set_xlim([self.x_min,self.x_max])
        self.ax.set_ylim([self.y_min,self.y_max])
        self.ax.scatter(0,0,c='r',s=100)
        self.ax.annotate("Mic",(0,0),size= self.text_size)
        
        self.scat = self.ax.scatter([], [], c= "orange",s=100)
        
        if self.save_video:
            return self.scat,
        
        else: 
            self.__annotations__= self.ax.annotate("",(0,0),size= self.text_size)
            self.__annotations1__= self.ax.annotate("",(0,0),size= self.text_size)
            self.__annotations2__= self.ax.annotate("",(0,0),size= self.text_size)
            self.__annotations3__= self.ax.annotate("",(0,0),size= self.text_size)
            
            
            #For FuncAnimation's sake, we need to return the artist we'll be using
            #Note that it expects a sequence of artists, thus the trailing comma.
            return self.scat,self.__annotations__,self.__annotations1__, self.__annotations2__,self.__annotations3__
        
    

    def data_stream(self):
        """Generate a random walk (brownian motion). Data is scaled to produce
        a soft "flickering" effect."""
        
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
        self.ax.set_title("the frame number: {}".format(f),size = self.text_size, loc= 'center')
        self.setup_plot()
        
        
        #print(x,y)
        # Set x and y data...
        self.scat.set_offsets(np.column_stack((x,y)))
        
        if self.save_video:
            for i in range(len(name)):
                self.annotaion = self.ax.annotate(name[i],(x[i],y[i]),size= self.text_size)
        else:
            if len(name) == 1:
                self.__annotations__= self.ax.annotate(name[0],(x[0],y[0]),size= self.text_size)
                return self.scat,self.__annotations__
            
            if len(name) == 2:
                self.__annotations__= self.ax.annotate(name[0],(x[0],y[0]),size= self.text_size)
                self.__annotations1__= self.ax.annotate(name[1],(x[1],y[1]),size= self.text_size)
                return self.scat, self.__annotations__, self.__annotations1__
            
            if len(name) == 3:
                self.__annotations__= self.ax.annotate(name[0],(x[0],y[0]),size= self.text_size)
                self.__annotations1__= self.ax.annotate(name[1],(x[1],y[1]),size= self.text_size)
                self.__annotations2__= self.ax.annotate(name[2],(x[2],y[2]),size= self.text_size)
                return self.scat, self.__annotations__, self.__annotations1__, self.__annotations2__
            
            if len(name) == 4:
                self.__annotations__= self.ax.annotate(name[0],(x[0],y[0]),size= self.size)
                self.__annotations1__= self.ax.annotate(name[1],(x[1],y[1]),size= self.size)
                self.__annotations2__= self.ax.annotate(name[2],(x[2],y[2]),size= self.size)
                self.__annotations3__= self.ax.annotate(name[3],(x[3],y[3]),size= self.size)
                return self.scat,self.__annotations__,self.__annotations1__, self.__annotations2__,self.__annotations3__
            
            
        # We need to return the updated artist for FuncAnimation to draw..
        # Note that it expects a sequence of artists, thus the trailing comma.
        return self.scat,
    


def main(argv):
    print(argv)
    option = False if len(argv) < 2 else argv[1]
    print(option)
    df = pd.read_csv("test2.csv")
    data = np.column_stack((df["Frames"].to_numpy(),df["X"].to_numpy(),df["Y"].to_numpy()))
    #st.write(data)
    name = df["Class Name"].to_numpy()
    start = timer()
    anme = Animation(data,name,option)
    print("without GPU:", timer()-start)
    

#test fi
if __name__ == '__main__':
    sys.exit(main(sys.argv))

