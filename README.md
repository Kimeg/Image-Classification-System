[Image Classification System]

-Objective: This project is intended to provide a full scale environment where image data generation and classifying model generation functionalities are available through tensorflow and/or pytorch API. As a developer of this project myself, I have had lots of learning opportunities during the implementation of customized methods from scratch. Although tensorflow and pytorch backends provide such utilities, I have decided to build them one by one with my own logics. 



-Image Data: 
> Description : Rectangle, triangle, and circle images are generated through data augmentation technique ( randomized variations in width, height, location, etc. ) 
> Size: ( these are default settings, please configure according to your needs. )
 [1] Training set (1,000)
 [2] Validation set (300)
 [3] Test set (200)



-Usage: 
> python run.py tf      ( for tensorflow backend)
> python run.py torch   ( for pytorch backend)



-Output:
> The prediction results will be printed on your terminal. 



-Note: 
> This codebase may be used for benchmarking purposes, such as model quality comparison between the two deep-learning frameworks or customization of image data ( you may need to implement such methods that satisfy your own needs ). 
> Existing image data and model files guarantee that the corresponding file generation steps are skipped for direct access to test dataset prediction.
