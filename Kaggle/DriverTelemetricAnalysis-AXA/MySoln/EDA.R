dirPath = 'G:/Continuing Education/Research & Presentations/Self - Machine Learning/Kaggle/DriverTelemetricAnalysis-AXA/'

trips = list.files("../drivers/1/")

i = 0
trip = list()

for(t in trips)
{
    t = read.csv(paste0(dirPath,"drivers/1/", t))
    trip[i][] = t
    i=i+1
}