def binaryToGrayScale(filePath, outputPath):
    f = open(filePath, 'rb') # Đọc tệp tin
    filename = os.path.basename(filePath)
    ln = os.path.getsize(filePath) # Tính độ dài bytes
    width = 256
    rem = ln % width
    a = array.array("B")
    a.fromfile(f, ln-rem) #chia bytes thành từng đơn vị 8 bytes
    f.close()
    g = numpy.reshape(a, (len(a) / width, width))
    g = numpy.unit8(g)
    scipy.misc.save(outputPath+filename+'.png', g) # Lưu ảnh