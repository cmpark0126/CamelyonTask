
def open_slide(file_name):

"
    open slide
    input file name
    return openslide object
"

def get_ROI_cord(openslide_obj):

"   
    get tumor ROI cordinate      
    input openslide obj
    return numpy list(x,y,width, height)
    By JunWooNim
"

def make_patch(openslide_obj, (x,y,width,height)):

"
    make a patch using cordinate
    input openslide_obj, cordination imformation about tumor
    return numpy list(patch)
"
