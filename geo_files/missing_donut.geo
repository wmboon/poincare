//+
SetFactory("OpenCASCADE");
Torus(1) = {0.5, 0.5, 0.5, 0.2, 0.1, 2 * Pi};
//+
Box(2) = {0, 0, 0, 1, 1, 1};
//+
Delete {
  Volume{1}; Volume{2}; 
}
//+
Surface Loop(3) = {7, 2, 4, 6, 3, 5};
//+
Surface Loop(4) = {1};
//+
Volume(1) = {3, 4};
//+
Physical Volume(15) = {1};
