# basic-classificator-py
Basic classificator program using Python and TensorFlow

Here is a little instruction for work with this code.
Menu in Russian! (sorry...)

## Before start:
#### Requirements: 
- Install:  
1 - Python 3.6.2  
2 - TensorFlow 1.10
- To use the GPU:  
3 - CUDA 9.0 (.176) (With GPU-ver TensorFlow)
- Python libraries:  
4 - PIL  
5 - Numpy  
6 - Pickle

#### About the paths to files and folders:
Currently, the program only works with a pre-prepared folder tree. Below is a list of required directories.
- C:\TensFlow
  - C:\TensFlow\Image_Borders
    - C:\TensFlow\Image_Borders\Input_Images
      - C:\TensFlow\Image_Borders\Input_Images\Datas
      - C:\TensFlow\Image_Borders\Input_Images\Maps
      - C:\TensFlow\Image_Borders\Input_Images\Ready
        - C:\TensFlow\Image_Borders\Input_Images\Ready\Test_img
        - C:\TensFlow\Image_Borders\Input_Images\Ready\Train_data
      - C:\TensFlow\Image_Borders\Input_Images\Trans
    - C:\TensFlow\Image_Borders\Output_Images
      - C:\TensFlow\Image_Borders\Output_Images\Binary_Format
      - C:\TensFlow\Image_Borders\Output_Images\Image_Format
    - C:\TensFlow\Image_Borders\Save_var
      - C:\TensFlow\Image_Borders\Save_var\vol_1
  
## Instruction:
The program processes images in the .bmp format, 360x240 px.  
Place the suitable images in the "Datas" folder, and then copy them into "Maps". Pictures with maps for learning should be modified: select the objects necessary for classification in absolute red color (FF0000).  
Now you can train the neural network. The size of the sliding window is currently fixed, is 28x28 pixels.  
Be careful! The code uses a lot of RAM!
