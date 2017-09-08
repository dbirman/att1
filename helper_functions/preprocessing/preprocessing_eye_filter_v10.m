function filtVal = preprocessing_eye_filter_v10(val,filtOrder,cutOffFreq)
% ----------------------------------------------------------------------
% filtVal = filterSacTrace(val,filtOrder,cutOffFreq)
% ----------------------------------------------------------------------
% Goal of the function :
% filter use to smooth the eye trace data for nice graphs
% ----------------------------------------------------------------------
% Input(s) :
% val = matrix of data to filter
% filtOrder = order of the filter fir1 used
% cutOffFreq = cutoff used by fr1 filter
% ----------------------------------------------------------------------
% Output(s):
% filtVal = values filtered
% ----------------------------------------------------------------------
% Expemple :
% filtVal = filterSacTrace(val,35,0.01)
% ----------------------------------------------------------------------
% Function created by Martin SZINTE (martin.szinte@gmail.com)
% Last update : 27 / 05 / 2011
% Project : RealMot Blank
% Version : 1.5
% ----------------------------------------------------------------------


filtVal = filtfilt(fir1(filtOrder,cutOffFreq),1,val);

end