#include"stdafx.h"
#include"MyFile.h"

///////////////////////////////////////
// ★★★ 读写文件二进制文件比txt快得多！！！
///////////////////////////////////////

using namespace cv;
using namespace std;

// 【文件后缀】返回文件后缀
string getFileSuffix(string filename)
{
	int where = (int)(filename.find_last_of('.'));
	string sub_str = filename.substr(where+1, filename.size() - where-1);      // string型的提取子串方法
	transform(sub_str.begin(), sub_str.end(), sub_str.begin(), ::tolower);
	return sub_str;
}

// 【文件后缀】返回文件名称 返回文件名 hello.txt 则返回hello
string getFileName(string filename)
{
	return CString2string(getFileName(string2CString(filename)));
}

// 【文件名】返回文件名 hello.txt 则返回hello
CString getFileName(CString fileName){
	CString strExt;
	int nPos = fileName.ReverseFind(_T('.'));
	if (nPos == -1)
		return _T("");

	strExt = fileName.Left(nPos);
	return strExt;
}

// 【查询文件】获取某一路径下的所有后缀为suffix的文件
CStringArray* findAllSpecFiles(CString path, CString suffix){
	CStringArray* result = new CStringArray;

	CFileFind finder;
	path += _T("\\*.") + suffix;
	BOOL bContinue = finder.FindFile(path);

	while (bContinue){
		bContinue = finder.FindNextFileW();
		result->Add(finder.GetFileName());
	}
	return result;
}

// 【查询文件】获取某一路径下的所有txt文件
CStringArray* findAllTxt(CString path){
	CStringArray* result = new CStringArray;

	CFileFind finder;
	path += _T("\\*.txt");
	BOOL bContinue = finder.FindFile(path);

	while (bContinue){
		bContinue = finder.FindNextFileW();
		result->Add(finder.GetFileName());
	}
	return result;
}

// 【查询文件】获取某一路径下的所有文件夹
CStringArray* findAllFolder(CString path){
	CStringArray* result = new CStringArray;

	CFileFind finder;
	path += _T("\\*.*");
	BOOL bContinue = finder.FindFile(path);

	while (bContinue){
		bContinue = finder.FindNextFileW();
		if (finder.IsDirectory() && !finder.IsHidden())
			result->Add(finder.GetFileName());
	}
	return result;
}

// 保存Mat到xml文件  filepath必须以.xml结尾 文件名不能为纯数字
bool saveMat(string filepath, Mat data){
	FileStorage fsFeature(filepath, FileStorage::WRITE);
	int index0 = (int)(filepath.rfind('.'));
	int index1 = (int)(filepath.rfind('\\', index0));
	int index1_ = (int)(filepath.rfind('/', index0));
	index1 = max(index1, index1_);
	if (index1 == -1)
		index1 = 0;
	fsFeature << filepath.substr(index1 + 1, index0 - index1-1) << data;
	fsFeature.release();
	return true;
}

// 【查询文件】 获取某一路径下的所有文件并保存至txt， saveFolder默认为true，该路径下的文件夹也存储进txt
bool saveAllFilePaths(string FolderPath, string savepath, bool saveFolder){
	ofstream hello(savepath);	// 此处为相对路径，也可以改为绝对路径 
	
	CFileFind finder;
	CString path = string2CString(FolderPath)+ _T("\\*.*");
	BOOL bContinue = finder.FindFile(path);

	while (bContinue){
		bContinue = finder.FindNextFileW();
		if (finder.IsDirectory()){
			if (!finder.IsHidden() && saveFolder)
				hello << CString2string(finder.GetFileName()) << endl;
		}
		else {
			hello << CString2string(finder.GetFileName()) << endl;
		}
		
	}
	
	
	hello.close();
	return true;
}


//  【读文件】每行作为一个字符串存入
std::vector<string> readTextInLines(std::string filepath){
	std::vector<std::string> configList;
	FILE *fp = fopen(filepath.c_str(), "r");
	if (fp){
		char temp[100];
		while (!feof(fp)){
			fscanf(fp, "%s", temp);
			configList.push_back(std::string(temp));
		}
		configList.pop_back();
		fclose(fp);
	}
	else {
		printf("---ERROR！！！---\n无法打开文件，请检查文件路径！！！\n");
	}
	return configList;
}

// 【读文件】从文件中读取下一个字段的数据，直到遇见下一个制表符（默认）或文件尾
// 首先清除string结果变量中以前的任何内容（这样可以在循环中更方便地使用函数，在循环中，同样的变量将被重复使用）。
// 然后保留一些空格，以避免频繁地重新定位。最后循环读取字符。
void get_chunk(istream& in, string& s, char terminator){
	s.erase(s.begin(), s.end());
	s.reserve(20);
	string::value_type  ch;
	while (in.get(ch) && ch != terminator)
		s.insert(s.end(), ch);
}

// 【读文件】每行作为一个字符串存入
bool readStringList(const string& filename, vector<string>& l){
	l.resize(0);
	FileStorage fs(filename, FileStorage::READ);
	if (!fs.isOpened())
		return false;
	FileNode n = fs.getFirstTopLevelNode();
	if (n.type() != FileNode::SEQ)
		return false;
	FileNodeIterator it = n.begin(), it_end = n.end();
	for (; it != it_end; ++it)
		l.push_back((string)*it);
	return true;
}

// 选择保存文件路径
string getSavePath(string dlgName, string suffix){
	// 设置过滤器
	string strFilter = "\"" + dlgName + "(*." + suffix + ")|*." + suffix + "|所有文件(*.*)|*.*||";
	//TCHAR szFilter[] = _T("标定结果(*.yml)|*.yml|所有文件(*.*)|*.*||");
	// 构造保存文件对话框   
	//CFileDialog fileDlg(FALSE, _T("yml"), _T("标定结果"), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter, NULL);
	CFileDialog fileDlg(FALSE, string2CString(suffix), string2CString(dlgName), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, pchar2pTCHAR(string2pChar(strFilter)), NULL);
	//CFileDialog fileDlg(FALSE, string2CString(suffix), string2CString(dlgName), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter, NULL);

	CString strFilePath;

	string filepath = "";

	// 显示打开文件对话框   
	if (IDOK == fileDlg.DoModal())
	{
		// 如果点击了文件对话框上的“打开”按钮，则将选择的文件路径显示到编辑框里   
		strFilePath = fileDlg.GetPathName();

		CStringA stra(strFilePath.GetBuffer(0));
		strFilePath.ReleaseBuffer();
		filepath = stra.GetBuffer(0);
		stra.ReleaseBuffer();
	}
	return filepath;
}

// 选择保存文件路径
bool getSavePath(string& filepath){
	// 设置过滤器
	TCHAR szFilter[] = _T("标定结果(*.yml)|*.yml|所有文件(*.*)|*.*||");
	// 构造保存文件对话框   
	CFileDialog fileDlg(FALSE, _T("yml"), _T("标定结果"), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT, szFilter, NULL);

	CString strFilePath;

	// 显示打开文件对话框   
	if (IDOK == fileDlg.DoModal())
	{
		// 如果点击了文件对话框上的“打开”按钮，则将选择的文件路径显示到编辑框里   
		strFilePath = fileDlg.GetPathName();

		CStringA stra(strFilePath.GetBuffer(0));
		strFilePath.ReleaseBuffer();
		filepath = stra.GetBuffer(0);
		stra.ReleaseBuffer();
		return true;
	}
	else
	{
		return false;
	}
}

// 从txt中读入2维点集
void readIntPoints2D(CString filename, vector<Point> &pts){
	FILE *fp;
	fopen_s(&fp, CString2pChar(filename), "r");

	Point tmp;
	for (int i = 0; !feof(fp); i++)
	{
		fscanf_s(fp, "%d %d\n", &tmp.x, &tmp.y, sizeof(float)); // 循环读
		pts.push_back(tmp);
	}
	fclose(fp);
}

// 【读文件】从txt中读取二维点集 实数
bool readRealPoints2D(CString filename, double* Point2D, long PointCount){
	fstream f1(LPCTSTR(filename), ios_base::in);
	if (!f1){
		AfxMessageBox(_T("警告：读入二维点集数据加载失败！"));
		return false;
	}
	for (int i = 1; i <= PointCount; i++){
		f1 >> *Point2D;
		Point2D++;
		f1 >> *Point2D;
		Point2D++;
	}
	f1.close();
	return true;
}

// 【读文件】从txt中读取三维点集 实数
bool readRealPoints3D(CString filename, double* Point3D, long PointCount){
	fstream f1(LPCTSTR(filename), ios_base::in);
	if (!f1){
		AfxMessageBox(_T("警告：读入二维点集数据加载失败！"));
		return false;
	}
	for (int i = 1; i <= PointCount; i++){
		f1 >> *Point3D;
		Point3D++;
		f1 >> *Point3D;
		Point3D++;
		f1 >> *Point3D;
		Point3D++;
	}
	f1.close();
	return true;
}

// 【写文件】写float*进入txt
void BmpSerialize(string filename, float *imgBuf, int dataSize){
	CFile saveF;
	if (FALSE == saveF.Open(string2CString(filename), CFile::modeCreate | CFile::modeWrite))
	{
		AfxMessageBox(_T("Serialize pic save error"));
	}
	CArchive ar(&saveF, CArchive::store);

	for (int i = 0; i<dataSize; i++)
	{
		ar << *(imgBuf + i);
	}

	ar.Close();
	saveF.Close();
}

// 【写文件】写float*进入txt
void writeFloatArrayInTxt(float* src, int len, std::string filepath, int offset){
	ofstream hello(filepath);	// 此处为相对路径，也可以改为绝对路径 
	for (int i = offset; i < offset+len; i++){
		hello << src[i] << endl;
	}
	hello.close();
	cout << "文件\"" << filepath << "\"保存成功!~" << endl;
}

// 【读文件】读txtMat
void readMatInTxt(Mat &data, std::string filepath){
	vector<double> d;
	FILE *fp;
	fopen_s(&fp, string2pChar(filepath), "r");

	double tmp;
	for (int i = 0; !feof(fp); i++)
	{
		fscanf_s(fp, "%lf\n", &tmp, sizeof(double)); // 循环读
		d.push_back(tmp);
	}
	fclose(fp);

	data = Mat::zeros(d.size(), 1, CV_64FC1);
	memcpy(data.ptr<double>(0), d.data(), sizeof(double)* d.size());
}

// 写Mat进入txt 
void writeMatInTxt(Mat src, std::string filepath){
	ofstream hello(filepath);	// 此处为相对路径，也可以改为绝对路径 
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.type() == CV_8UC1 || src.type() == CV_8U){
				int temp = src.at<uchar>(i, j);
				hello << temp << " ";
			}				
			else if (src.type() == CV_32FC1 || src.type() == CV_32F)
				hello << src.at<float>(i, j) << " ";
			else if (src.type() == CV_64FC1 || src.type() == CV_64F)
				hello << src.at<double>(i, j) << endl;
			else if (src.type() == CV_64FC2)
				hello << src.ptr<double>(0)[(i*src.cols + j) * 2] << " " << src.ptr<double>(0)[(i*src.cols + j) * 2+1] << endl;
		}
		//hello << endl;
	}
	hello.close();
}

// 写vector<Point2f>进入txt
void writePts2DInTxt(vector<Point2f> src, std::string filepath){
	ofstream hello(filepath);	// 此处为相对路径，也可以改为绝对路径 
	vector<Point2f>::iterator it = src.begin();

	for (; it<src.end(); it++)
	{
		
		hello << (*it).x << " " << (*it).y << " " << endl;
	}
	hello.close();
}

// 写vector<Point3d>进入txt
void writePts3DInTxt(vector<Point3d> src, std::string filepath){
	ofstream hello(filepath);	// 此处为相对路径，也可以改为绝对路径 
	vector<Point3d>::iterator it = src.begin();

	for (; it<src.end(); it++)
	{
		hello << (*it).x << " " << (*it).y << " " << (*it).z << " " << endl;
	}
	hello.close();
}

// 写vector<Point>进入txt
void writePts2DInTxt(vector<Point> src, std::string filepath){
	ofstream hello(filepath);	// 此处为相对路径，也可以改为绝对路径 
	vector<Point>::iterator it = src.begin();

	for (; it<src.end(); it++)
	{
		hello << (*it).x << " " << (*it).y << " " << endl;
	}
	hello.close();
}

// 【文件夹】选择一个文件夹
CString selectFolder(){
	TCHAR           szFolderPath[MAX_PATH] = { 0 };
	CString         strFolderPath = TEXT("");

	BROWSEINFO      sInfo;
	::ZeroMemory(&sInfo, sizeof(BROWSEINFO));
	sInfo.pidlRoot = 0;
	sInfo.lpszTitle = _T("请选择一个文件夹：");
	sInfo.ulFlags = BIF_DONTGOBELOWDOMAIN | BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE | BIF_EDITBOX;
	sInfo.lpfn = NULL;

	// 显示文件夹选择对话框  
	LPITEMIDLIST lpidlBrowse = ::SHBrowseForFolder(&sInfo);
	if (lpidlBrowse != NULL)
	{
		// 取得文件夹名  
		if (::SHGetPathFromIDList(lpidlBrowse, szFolderPath))
		{
			strFolderPath = szFolderPath;
		}
	}
	if (lpidlBrowse != NULL)
	{
		::CoTaskMemFree(lpidlBrowse);
	}
	system("cls");
	return strFolderPath;
}

// 【文件夹】判断文件夹是否存在
bool isFolderValidate(CString folderpath){
	return PathIsDirectory(folderpath);
}

// 【文件夹】判断文件夹是否存在
bool isFolderValidate(string folderpath){
	CString tmp = string2CString(folderpath);
	return isFolderValidate(tmp);
}

// 【文件】判断文件是否存在
bool isFileExist(string filepath)
{
	return (!_access(filepath.c_str(), 0));
}

// 【读取】读取矩阵
cv::Mat readMat(std::string filepath)
{
	Mat result;
	FileStorage fsRead(filepath, FileStorage::READ);

	int index0 = filepath.rfind('.');
	int index1 = filepath.rfind('\\', index0);
	int index1_ = filepath.rfind('/', index0);
	index1 = max(index1, index1_);
	if (index1 == -1)
		index1 = 0;

	fsRead[filepath.substr(index1 + 1, index0 - index1 - 1)] >> result;
	return result;
}

// 保存Mat到二值文件
cv::Mat readMatInBinaryFloat(std::string filename)
{
	size_t sz = getFileSize(filename);

	FILE *saveF;
	saveF = fopen(filename.c_str(), "rb");
	
	if (saveF == NULL)
	{
		std::printf("载入二值文件到Mat出错!\n");
		return cv::Mat();
	}
	int width, height, nChannels;
	fread(&width, sizeof(int), 1, saveF);
	fread(&height, sizeof(int), 1, saveF);
	fread(&nChannels, sizeof(int), 1, saveF);
	printf("readMatInBinaryFloat %d %d %d\n", width, height, nChannels);
	int cnt = width * height * nChannels;

	int sz_float = cnt * sizeof(float) + 12;
	int sz_double = cnt * sizeof(double) + 12;
	cv::Mat mat;
	
	// 说明是float文件
	if (sz_float == sz)
	{
		printf("file type: float\n");
		if (nChannels == 1)
			mat = cv::Mat::zeros(height, width, CV_32FC1);
		else if (nChannels == 3)
			mat = cv::Mat::zeros(height, width, CV_32FC3);

		float *data = (float*)mat.data;
		for (int i = 0; i < cnt; i++)
			fread(&data[i], sizeof(float), 1, saveF);
	}
	else if (sz_double == sz)
	{
		printf("file type: double\n");
		if (nChannels == 1)
			mat = cv::Mat::zeros(height, width, CV_64FC1);
		else if (nChannels == 3)
			mat = cv::Mat::zeros(height, width, CV_64FC3);

		double *data = (double*)mat.data;
		for (int i = 0; i < cnt; i++)
			fread(&data[i], sizeof(double), 1, saveF);

		if (nChannels == 1)
			mat.convertTo(mat, CV_32FC1);
		else if (nChannels == 3)
			mat.convertTo(mat, CV_32FC3);
	}

	fclose(saveF);
	return mat;
}


// 读取txt第一行并存入string
string readFstLineInTxt(string filepath){
	string str;
	ifstream fin(filepath.c_str(), ios::in);
	if (!fin){
		AfxMessageBox(_T("不能打开此文件！"));
		return "";
	}
	getline(fin, str);
	return str;
}

// 利用winWin32_FIND_DATA 读取文件夹下的文件名
void readImgNamefromFile(char *fileName, vector<string> &imgNames){
	// vector清零，参数设置
	imgNames.clear();
	WIN32_FIND_DATA file;
	int i = 0;
	char tempFilePath[MAX_PATH + 1];
	char tempFileName[50];
	// 转换输入文件名
	sprintf(tempFilePath, "%s/*", fileName);
	// 查找待操作文件的相关属性，读取到WIN32_FIND_DATA

	HANDLE handle = FindFirstFile(pchar2LPCWSTR(tempFilePath), &file);
	if (handle != INVALID_HANDLE_VALUE){
		FindNextFile(handle, &file);
		FindNextFile(handle, &file);
		// 循环便利得到文件夹的所有文件名
		do{
			sprintf(tempFileName, "%s\\", fileName);
			CString suffix = RightCString(file.cFileName, 3);
			if (suffix != _T("png") && suffix != _T("jpg") && suffix != _T("bmp") && suffix != "PNG" && suffix != "BMP" && suffix != "JPG")
				continue;

			imgNames.push_back(CString2string(file.cFileName));
			imgNames[i].insert(0, tempFileName);
			i++;
		} while (FindNextFile(handle, &file));
	}
	FindClose(handle);
}



// 【文件重命名】
// 给folderPath文件夹中的所有文件，增加前缀prefix
// useOld 是否使用原来的名字，如果是，则改后的文件只是加前缀否则就重新命名 为前缀后面加数字编号
bool renameFilesInFolder(string folderPath, string prefix, bool useOld/* = true*/,string suffix/* = ""*/){
	ofstream hello(folderPath);	// 此处为相对路径，也可以改为绝对路径 

	CFileFind finder;
	CString path = string2CString(folderPath) + _T("\\*.*");
	BOOL bContinue = finder.FindFile(path);

	int i = 0;

	while (bContinue){
		bContinue = finder.FindNextFileW();
		string name = CString2string(finder.GetFileName());
		string new_suffix = getFileSuffix(name);
		if (suffix != "")
			new_suffix = suffix;
		string new_name;

		if (!finder.IsDirectory()){
			
			if (useOld){
				new_name = prefix + "_" + name + "." + new_suffix;
			}
			else {
				new_name = prefix + int2string(i) + "." + new_suffix;
				//cout << "i = " << i << endl;
				i++;
			}
			//cout << new_name << endl;
			name = folderPath + "\\" + name;
			new_name = folderPath + "\\" + new_name;
			rename(name.c_str(), new_name.c_str());
		}
	}

	hello.close();
	return true;
}

// 【清空】清空文件
void clearFile(string filepath){
	ofstream file(filepath, ios::out);
	file.close();
} 

// 【读文件】STL 二进制形式
bool ReadBinary(string cfilename, vector<float>& coorX, vector<float>& coorY, vector<float>& coorZ,
	vector<float>& normX, vector<float>& normY, vector<float>& normZ)
{
	coorX.clear();
	coorY.clear();
	coorZ.clear();

	normX.clear();
	normY.clear();
	normZ.clear();

	char str[80];
	ifstream in;

	in.open(cfilename, ios::in | ios::binary);

	if (!in)
		return false;

	in.read(str, 80);

	//number of triangles  
	int unTriangles;
	in.read((char*)&unTriangles, sizeof(int));
	cout << "三角形数量: " << unTriangles << endl;

	if (unTriangles == 0)
		return false;

	// 没有给出法向量？？
	for (int i = 0; i < unTriangles; i++)
	{
		float coorXYZ[12];
		in.read((char*)coorXYZ, 12 * sizeof(float));
		
		normX.push_back(coorXYZ[0]);
		normY.push_back(coorXYZ[1]);
		normZ.push_back(coorXYZ[2]);
		
		for (int j = 1; j <= 3; j++)
		{
			coorX.push_back(coorXYZ[j * 3]);
			coorY.push_back(coorXYZ[j * 3+1]);
			coorZ.push_back(coorXYZ[j * 3+2]);
		}
		//cout << coorXYZ[0] << " " << coorXYZ[1] << " " << coorXYZ[2] << endl;
		in.read((char*)coorXYZ, 2);
	}
	in.close();

	//cout << coorX.size() / 3 << " triangles." << endl;
	return true;
}

// 【读文件】STL ASCⅡ码形式
bool ReadASCII(string cfilename)
{
	std::vector<float> coorX;
	std::vector<float> coorY;
	std::vector<float> coorZ;

	int i = 0, j = 0, cnt = 0, pCnt = 4;
	char a[100];
	char str[100];
	double x = 0, y = 0, z = 0;

	ifstream in;
	in.open(cfilename, ios::in);
	if (!in)
		return false;
	do
	{
		i = 0;
		cnt = 0;
		in.getline(a, 100, '\n');
		while (a[i] != '\0')
		{
			if (!islower((int)a[i]) && !isupper((int)a[i]) && a[i] != ' ')
				break;
			cnt++;
			i++;
		}

		while (a[cnt] != '\0')
		{
			str[j] = a[cnt];
			cnt++;
			j++;
		}
		str[j] = '\0';
		j = 0;

		if (sscanf(str, "%lf%lf%lf", &x, &y, &z) == 3)
		{
			coorX.push_back(x);
			coorY.push_back(y);
			coorZ.push_back(z);
		}
		pCnt++;
	} while (!in.eof());

	//  cout << "******  ACSII FILES　******" << endl;  
	//  for (int i = 0; i < coorX.size();i++)  
	//  {  
	//      cout << coorX[i] << " : " << coorY[i] << " : " << coorZ[i] << endl;  
	//  }  

	cout << coorX.size() / 3 << " triangles." << endl;
	return true;
}

// 【读文件】读STL文件 返回值是三角面片的数量
bool ReadSTLFile(string cfilename, vector<float>& coorX, vector<float>& coorY, vector<float>& coorZ,
	vector<float>& normX, vector<float>& normY, vector<float>& normZ)
{
	if (cfilename.empty())
		return false;

	ifstream in(cfilename, ios::in);
	if (!in)
		return false;

	//string headStr;
	//
	////for (int i = 0; i < 100; i++){
	//	getline(in, headStr, ' ');
	//	cout << headStr << endl;
	//}

	//
	//in.close();

	//if (headStr.empty())
	//	return false;

	/*if (headStr[0] == 's')
	{
		cout << "ASCII File." << endl;
		ReadASCII(cfilename);
	}
	else
	{*/
		cout << "Binary File." << endl;
		ReadBinary(cfilename, coorX, coorY, coorZ, normX, normY, normZ);
	//}
	return true;
}

// 【文件夹】创建文件夹
void createFolder(string folder){
	_mkdir(string2pChar(folder));
}

// 【文件夹】删除文件夹
void deleteFolder(string folder){
	system(string2pChar("rd /S /Q " + folder));
}

// 【剪贴板】复制文件到剪贴板
// 路径中的斜杠用反斜杠\ 而且得是俩杠\\ ，比如 f:\\3.bmp
int CopyFileToClipboard(string filename)
{
	const char* szFileName = string2pChar(filename);
	UINT uDropEffect;
	HGLOBAL hGblEffect;
	LPDWORD lpdDropEffect;
	DROPFILES stDrop;
	HGLOBAL hGblFiles;
	LPSTR lpData;
	uDropEffect = RegisterClipboardFormat(_T("Preferred DropEffect"));
	hGblEffect = GlobalAlloc(GMEM_ZEROINIT | GMEM_MOVEABLE | GMEM_DDESHARE, sizeof(DWORD));
	lpdDropEffect = (LPDWORD)GlobalLock(hGblEffect);
	*lpdDropEffect = DROPEFFECT_COPY;//复制; 剪贴则用DROPEFFECT_MOVE
	GlobalUnlock(hGblEffect);
	stDrop.pFiles = sizeof(DROPFILES);
	stDrop.pt.x = 0;
	stDrop.pt.y = 0;
	stDrop.fNC = FALSE;
	stDrop.fWide = FALSE;
	hGblFiles = GlobalAlloc(GMEM_ZEROINIT | GMEM_MOVEABLE | GMEM_DDESHARE, \
		sizeof(DROPFILES)+strlen(szFileName) + 2);
	lpData = (LPSTR)GlobalLock(hGblFiles);
	memcpy(lpData, &stDrop, sizeof(DROPFILES));
	strcpy(lpData + sizeof(DROPFILES), szFileName);
	GlobalUnlock(hGblFiles);
	OpenClipboard(NULL);
	EmptyClipboard();
	SetClipboardData(CF_HDROP, hGblFiles);
	SetClipboardData(uDropEffect, hGblEffect);
	CloseClipboard();
	return 1;
}

// 通过对话框选择保存路径 保存文件对话框
// filetype 是文件类型 比如 图片 文本 视频 音频等
// suffix 文件的后缀
// defaultName 是文件的默认名称
std::string selectSavePath(std::string filetype, std::string suffix, std::string defaultName/* = "output"*/){
	if (!suffix.empty()){
		defaultName += "." + suffix;
	}
	std::string szFilter = filetype + "(*." + suffix + ")|*." + suffix + "|所有文件(*.*)|*.*||";

	CFileDialog openFileDlg(FALSE, 0, string2CString(defaultName), OFN_HIDEREADONLY | OFN_OVERWRITEPROMPT,
		LPCTSTR(string2CString(szFilter).GetBuffer()), NULL);
	INT_PTR result = openFileDlg.DoModal();
	CString filePath;
	if (result == IDOK) {
		filePath = openFileDlg.GetPathName();
		return CString2string(filePath);
	}
	else{
		return "";
	}
}

size_t getFileSize(const std::string &file_name)
{
	std::ifstream in(file_name.c_str());
	in.seekg(0, std::ios::end);
	size_t size = in.tellg();
	in.close();
	return size;
}

// 保存Mat到二值文件
void saveMatInBinaryFloat(const cv::Mat& mat, std::string filename)
{
	FILE* saveF;
	saveF = fopen(filename.c_str(), "wb");
	if (saveF == NULL)
	{
		std::printf("保存Mat到二值文件出错!\n");
		return;
	}
	int width = mat.cols;
	int height = mat.rows;
	int nChannels = mat.channels();
	printf("%s  -- %d %d %d\n", filename.c_str(), width, height, nChannels);
	fwrite(&width, sizeof(int), 1, saveF);
	fwrite(&height, sizeof(int), 1, saveF);
	fwrite(&nChannels, sizeof(int), 1, saveF);

	int cnt = width * height * nChannels;
	float* data = (float*)mat.data;
	for (int i = 0; i < cnt; i++)
		fwrite(&data[i], sizeof(float), 1, saveF);
	fclose(saveF);
}