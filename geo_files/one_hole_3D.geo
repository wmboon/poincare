h = 0.10;
//+
SetFactory("OpenCASCADE");
Box(1) = {0, 0, 0, 1, 1, 1};
//+
Box(2) = {0.4, .4, .4, .2, .2, .2};
//+//+
Delete {
  Volume{1}; Volume{2}; 
}
//+
Surface Loop(3) = {6, 1, 3, 5, 2, 4};
//+
Surface Loop(4) = {12, 7, 9, 11, 8, 10};
//+
Volume(1) = {3, 4};
//+
Surface Loop(5) = {11, 7, 9, 12, 8, 10};
//+
Physical Volume(25) = {1};
