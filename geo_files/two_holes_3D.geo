h = 0.15;
//+
Point(1) = {0, 0, 0, h};
//+
Point(2) = {1, 0, 0, h};
//+
Line(1) = {1, 2};
//+
Point(5) = {0.225, 0.625, 0, h};
//+
Point(6) = {0.375, 0.625, 0, h};
//+
Line(5) = {5, 6};
//+

//+
Point(9) = {0.625, 0.225, 0, h};
//+
Point(10) = {0.775, 0.225, 0, h};
//+
Line(9) = {9, 10};
//+
//+
Extrude {0, 0.15, 0} {
  Point{6}; Point{5}; Point{9}; Point{10}; 
}
//+
Line(14) = {12, 11};
//+
Line(15) = {13, 14};

//+
Extrude {0, 1, 0} {
  Point{1}; Point{2}; 
}
//+
Line(18) = {15, 16};
//+
Curve Loop(1) = {18, -17, -1, 16};
//+
Curve Loop(2) = {14, -10, -5, 11};
//+
Curve Loop(3) = {15, -13, -9, 12};
//+
//+
Extrude {0, 0, 1} {
  Point{15}; Curve{18}; Point{16}; Curve{17}; Point{2}; Curve{1}; Point{1}; Curve{16}; Curve{11}; Point{12}; Point{5}; Curve{5}; Point{6}; Curve{10}; Curve{14}; Point{11}; Point{13}; Curve{15}; Curve{12}; Point{9}; Curve{9}; Curve{13}; Point{10}; Point{14}; 
}
//+
Curve Loop(4) = {32, 20, -24, -28};
//+
Curve Loop(5) = {36, 48, -44, -40};
//+
Curve Loop(6) = {57, 53, -65, -61};
//+
Plane Surface(69) = {4, 5, 6};
//+
Plane Surface(70) = {1, 2, 3};
//+
Surface Loop(1) = {23, 70, 27, 69, 35, 31, 43, 47, 51, 39, 64, 68, 56, 60};
//+
Volume(1) = {1};
//+
Physical Volume("vol", 71) = {1};
