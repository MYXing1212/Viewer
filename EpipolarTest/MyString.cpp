#include"stdafx.h"
//#include"stdafx.h"
#include "MyString.h"

using namespace cv;

string ComputeType[9] = {"^", "sin", "cos", "tan", "log","+", "-", "*", "/"};

MyString::MyString(void)
{
}


MyString::~MyString(void)
{
}



// const char* ת char*
char* pConstC2pC(const char* ch)
{
	char *buf = new char[strlen(ch) + 1];
	strcpy_s(buf,strlen(buf), ch);
	return buf;
}

//// stringתLPCSTR
//wchar_t* string2LPCSTR(string str)
//{
//	const char* tmp = str.c_str();
//	return 	this->Char2LPCTSTR(this->pConstC2pC(tmp));
//}

// �ַ��ָ��
void MyString::splitString(string input, string flag, vector<string>& result)
{
	if (flag.empty()) return ;

	// find_first_of��input, index�� input��Ҫ�������ַ�����index����������ʼλ��
	size_t start = 0, index = input.find_first_of(flag, 0);

	// string �ཫ npos ����Ϊ��֤�����κ���Ч�±��ֵ��
	while (index != input.npos)
	{
		if (start != index) {
			result.push_back(input.substr(start, index - start));
		}
		start = index + 1;
		index = input.find_first_of(flag, start);
	}
	if (!input.substr(start).empty()) {
		result.push_back(input.substr(start));
	}
}

// �ַ���ת����
long MyString::str2long(string s) {
	long num;
	stringstream ss(s);
	ss >> num;
	return num;
}

// �ָ��ַ�������õ�һ�ηָ���
void MyString::getFirstSplit(string input, string flag, string& result) {
	vector<string> tmp;
	splitString(input, flag, tmp);
	result = tmp[0];
}

// �ָ��ַ�����������һ�ηָ���
void MyString::getLastSplit(string input, string flag, string& result) {
	vector<string> tmp;
	splitString(input, flag, tmp);
	result = tmp[tmp.size()-1];
}

// ��char*�����ת��Ϊ��CString����ʽ
//  �������ص���Ϣ
// char *buf_recv = "Hello world��"
// CString parse_str;
// parse_str.Format(_T("%s"), CStringW(buf_recv));

// ��char*�ַ���ת��ΪLPCTSTR�ַ���
wchar_t* Char2LPCTSTR(char * str)
{
	int num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
	wchar_t *wide = new wchar_t[num];
	MultiByteToWideChar(0, 0, str, -1, wide, num);
	return wide;
}




char *w2c(char *pcstr, const wchar_t *pwstr, size_t len)

{
	int nlength = (int)wcslen(pwstr);
	//��ȡת����ĳ���
	int nbytes = WideCharToMultiByte(0, // specify the code page used to perform the conversion
		0,         // no special flags to handle unmapped characters
		pwstr,     // wide character string to convert
		nlength,   // the number of wide characters in that string
		NULL,      // no output buffer given, we just want to know how long it needs to be
		0,
		NULL,      // no replacement character given
		NULL);    // we don't want to know if a character didn't make it through the translation

	// make sure the buffer is big enough for this, making it larger if necessary
	if (nbytes>(int)len)   nbytes = (int)len;
	// ͨ�����ϵõ��Ľ����ת��unicode �ַ�Ϊascii �ַ�
	WideCharToMultiByte(0, // specify the code page used to perform the conversion
		0,         // no special flags to handle unmapped characters
		pwstr,   // wide character string to convert
		nlength,   // the number of wide characters in that string
		pcstr, // put the output ascii characters at the end of the buffer
		nbytes,                           // there is at least this much space there
		NULL,      // no replacement character given
		NULL);
	return pcstr;
}

// �ж�һ���ַ����Ƿ�Ϊ�����ַ���
bool MyString::ifHuiwenString(string a)
{
	int num = (int)a.length();
	bool flag = true;
	for (int i = 0; i < num; i++)
	{
		if (a.substr(num - 1 - i, 1) != a.substr(i, 1))   // ��Ӧλ�ò�ͬ
		{
			flag = false;
			return flag;
		}
	}
	return flag;
}

// intת CString
CString int2CString(int i){
	CString str;
	str.Format(_T("%d"), i);
	return str;
}

// floatת CString
CString float2CString(float f){
	CString str;
	str.Format(_T("%f"), f);
	return str;
}

// doubleת CString
CString double2CString(double d){
	CString str;
	str.Format(_T("%lf"), d);
	return str;
}

// ��stringתconst char*�ַ���
const char* string2pChar(string str){
	CString tmp = string2CString(str);
	return CString2pChar(tmp);
}

// ��CString�ַ���ת��Ϊconst char*�ַ���
const char* CString2pChar(CString cstring)
{
#ifdef UNICODE
	DWORD dwNum = WideCharToMultiByte(CP_OEMCP, NULL, cstring.GetBuffer(0), -1, NULL, 0, NULL, FALSE);
	char *psText;
	psText = new char[dwNum];
	if (!psText)
		delete[]psText;
	WideCharToMultiByte(CP_OEMCP, NULL, cstring.GetBuffer(0), -1, psText, dwNum, NULL, FALSE);
	return (const char*)psText;
#else
	return (LPCTSRT)cstring;
#endif
}

// ��CString�ַ���ת��Ϊchar* �ַ���
char* CString2pchar(CString cstring){
	char *pNumber = new char[64];
	strcpy(pNumber, CString2pChar(cstring));
	return pNumber;
}

// ��CString�ַ���ת��Ϊstring�ַ���
string CString2string(CString cstring)
{
	CStringA stra(cstring.GetBuffer(0));
	cstring.ReleaseBuffer();
	std::string result = stra.GetBuffer(0);
	stra.ReleaseBuffer();
	return result;
}

BOOL MyString::WChar2MByte(LPCWSTR srcBuff, LPSTR destBuff, int nlen)
{
	int n = 0;
	n = WideCharToMultiByte(CP_OEMCP, 0, srcBuff, -1, destBuff, 0, 0, FALSE);
	if (n < nlen)
		return FALSE;
	WideCharToMultiByte(CP_OEMCP, 0, srcBuff, -1, destBuff, nlen, 0, FALSE);
	return TRUE;
}

/***********************************************************
** ��������: int ConvertUnicodeToAscii(const CString &csData, char* pszData)

** ��������: Unicodeת���ֽ�

** ��    ��: const CString &csData Unicode�ַ���
char* pszData ����ת���Ķ��ֽ��ַ���

** �� �� ֵ: ת���ĸ���
************************************************************/
int MyString::ConvertUnicodeToAscii(const CString &csData, char* pszData)
{
	int nDataLength;
	nDataLength = WideCharToMultiByte(CP_ACP, 0, csData, -1, NULL, 0, NULL, FALSE);
	WideCharToMultiByte(CP_ACP, 0, csData, -1, pszData, nDataLength, NULL, 0);
	pszData[nDataLength - 1] = '\0';

	return nDataLength - 1;
}

CString MyString::getTimeCString()
{
	CTime time = CTime::GetCurrentTime();
	return time.Format(_T("%Y/%m/%d %H:%M:%S "));
}

// ����ת�ַ�
string int2string(int a){
	CString tmp;
	tmp.Format(_T("%d"), a);
	string result = CString2string(tmp);
	return result;
}

// CString ת��Ϊ int
int CString2int(CString str){
	return _ttoi(str);
}

// ������ת�ַ�
string double2string(double a, int accuracy){
	CString tmp;
	CString digit;
	digit.Format(_T("%d"), accuracy);
	tmp.Format(_T("%.")+digit+_T("lf"), a);
	return CString2string(tmp);
}

// ��string�ַ���ת��ΪCString�ַ���
CString string2CString(string str){
	return (CString)str.c_str();
}

// ��char* ת��Ϊ LPCWSTR
LPCWSTR pchar2LPCWSTR(char* p){
	//WCHAR wstr[MAX_PATH] = { 0 };
	WCHAR* wstr = (WCHAR*)malloc(sizeof(WCHAR)*MAX_PATH);
	MultiByteToWideChar(CP_ACP, 0, p, -1, wstr, sizeof(wstr));
	return wstr;
}


// ��char*�ַ���ת��ΪLPCTSTR�ַ���
LPCTSTR pchar2LPCTSTR(char * str)
{
	int num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
	wchar_t *wide = new wchar_t[num];
	MultiByteToWideChar(0, 0, str, -1, wide, num);
	return wide;
}

// ��char*תwchar_t*
wchar_t* pchar2pwchar_t(char* str){
	int num = MultiByteToWideChar(0, 0, str, -1, NULL, 0);
	wchar_t *wide = new wchar_t[num];
	MultiByteToWideChar(0, 0, str, -1, wide, num);
	return wide;
}

// char* ת TCHAR*
TCHAR* pchar2pTCHAR(const char* str){	
	int iLength;
	iLength = MultiByteToWideChar(CP_ACP, 0, str, (int)strlen(str) + 1, NULL, 0);
	TCHAR* tchar = (TCHAR*)malloc(sizeof(TCHAR)*iLength);
	MultiByteToWideChar(CP_ACP, 0, str, (int)strlen(str) + 1, tchar, iLength);
	return tchar;
}

// ȡCString���ҵ�count���ַ�
CString RightCString(CString input, int count){
	return input.Right(count);
}

// CString�Ӻ���ǰ���ַ�'x',���ڵ�λ�� 2\1.txt ,'\\' ���Ϊ1
int b2fFind(CString src, char c){
	TCHAR* w = (TCHAR*)malloc(sizeof(TCHAR)* 1);
	w = pchar2pTCHAR(&c);
	return src.ReverseFind(w[0]);
}

// ȡCString�����count���ַ�
CString LeftCString(CString input, int count){
	return input.Left(count);
}

// ȡCString�Ӻ���ǰ�ҡ�x�� ֮ǰ���ַ�
CString LStr_b2fFind(CString input, char c){
	return LeftCString(input, b2fFind(input, c));
}

// ȡCString�Ӻ���ǰ�ҡ�x�� ֮����ַ�
CString RStr_b2fFind(CString input, char c){
	UINT uStrLength = input.GetLength();
	return RightCString(input, uStrLength - b2fFind(input, c) - 1);
}

// ����ӡ�ַ�������ӡCStingArray
void printCStringArray(CStringArray* strArray){
	for (int i = 0; i < strArray->GetSize(); i++){
		//AfxMessageBox(strArray->GetAt(i));
		std::cout << CString2pChar(strArray->GetAt(i)) << endl;
	}
}

// ���ַ�����ת��Ϊdouble
double string2double(string str){
	return atof(string2pChar(str));
}

// ���ַ�����ת��Ϊbool
bool string2bool(string str){
	if(str == "1" || str == "true")
		return true;
	else
		return false;
}

// ���ַ�����ת��Ϊint
int string2int(string str){
	return atoi(string2pChar(str));
}


// ��·������ȡ����·��
CString getWorkDir(){
	TCHAR exeFullPath[MAX_PATH];
	GetModuleFileName(NULL, exeFullPath, MAX_PATH);
	CString workPath = (CString)exeFullPath;
	//AfxMessageBox(workPath);
	return (LStr_b2fFind(workPath, '\\'));
}

// ����ǰʱ�䡿
CString getCurrentTimeString(bool bAbbrev){
	SYSTEMTIME sys;
	GetLocalTime(&sys);

	CString timeStr;
	
	if (!bAbbrev)
		timeStr.Format(_T("%04d��%02d��%02d��  %02d:%02d:%02d"), sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond);
	else 
		timeStr.Format(_T("%04d%02d%02d_%02d%02d%02d"), sys.wYear, sys.wMonth, sys.wDay, sys.wHour, sys.wMinute, sys.wSecond);
	
	return timeStr;
}

// ����ǰ���ڡ�
CString getCurrentDateString(){
	SYSTEMTIME sys;
	GetLocalTime(&sys);

	CString timeStr;
	timeStr.Format(_T("%04d%02d%02d"), sys.wYear, sys.wMonth, sys.wDay);

	return timeStr;
}

// ɾ�������ַ�a
bool removeAllSpecChar(string& str, char c){
	string::iterator new_end = remove_if(str.begin(), str.end(), bind2nd(equal_to<char>(), c));
	str.erase(new_end, str.end());
	return true;
}

// �ҵ���Ե�����
int findMatchBracket(string str, int index){
	CV_Assert(str[index] == '(' || str[index] == ')');
	if (str[index] == '('){
		int tmpCount = 0;
		for (int i = index + 1; i < str.length(); i++){
			if (str[i] == '(')
				tmpCount++;
			if (str[i] == ')'){
				if (tmpCount == 0)
					return i;
				else
					tmpCount--;
			}
		}
		return -1;
	}
	else if(str[index] == ')'){
		int tmpCount = 0;
		for (int i = index - 1; i >= 0; i--){
			if (str[i] == ')')
				tmpCount++;
			if (str[i] == '('){
				if (tmpCount == 0)
					return i;
				else
					tmpCount--;
			}
		}
		return -1;
	}
}


// �����Ҳ��С�����ڵ����� 
bool simplifyRightestBrackets(string& str, string& minus_mask){
	//std::cout << "--> simplifyRightestBrackets" << endl;
	int index_left = (int)str.find_last_of("(");
	if (index_left == -1)
		return false;
	int index_right = (int)str.find_first_of(")", index_left);
	//std::cout << index_left << endl << index_right << endl;

	string content = str.substr(index_left + 1, index_right - index_left - 1);
	//cout << "content = " << content << endl;
	string tmp_mask = minus_mask.substr(index_left + 1, index_right - index_left - 1);
	//string tmp_mask(content.length(), '0');
	if (content[0] == '-')
		tmp_mask[0] = '1';
	//cout <<"tmp_mask = "<< tmp_mask << endl;

	string result = double2string(calcEquationWithoutBrackets(content, tmp_mask), 15);
	str.replace(index_left, index_right - index_left + 1,result);

	string tmp_str(result.length(), '0');
	if (result[0] == '-')
		tmp_str[0] = '1';
	minus_mask.replace(index_left, index_right - index_left + 1, tmp_str);	

	//std::cout << str << endl << endl;
	return true;
}

// ��û�����ŵ�ʽ����ֵ
double calcEquationWithoutBrackets(string content, string& minus_mask){
	int index_exp = 0;

	simplifyEquation(content, "^", minus_mask);
	simplifyEquation(content, "spe", minus_mask);
	simplifyEquation(content, "*/", minus_mask);
	simplifyEquation(content, "+-", minus_mask);

	//std::cout << "calcEquationWithoutBrackets:" << endl << "result = " << content << endl << endl;;
	return string2double(content);
}

// �ж��Ƿ��ǵ���ʽ
bool isMonomial(string input, vector<string> alph){
	return true;
}


// ������
string baseDerivation(string input, string flag){
	CV_Assert(flag != "");
	int index = (int)input.find_first_of(flag);
	int n = (int)input.size();

	string result = "";
	if (index == -1){
		result = "0";
		return result;
	}
	else {
		int tmpIndex;
		
		// ��sinx��
		if ((tmpIndex = (int)input.find("sin")) != -1){
			string back = input.substr(tmpIndex + 3, n - tmpIndex - 3);
			if (back == flag){
				result = "cos" + flag;
				return result;
			}
			else if (back != flag){
				result = "Error!";
				return result;
			}
		}
		// ��a^x����x^a��
		if ((tmpIndex = (int)input.find("^")) != -1){
			string pre = input.substr(0, tmpIndex);
			string back = input.substr(tmpIndex + 1, n - tmpIndex);
			if (pre == flag){
				result += back + "*" + flag + "^" + baseCompute(back, "1.0", "-");
				return result;
			}
			else if (back == flag){
				if (pre == "e"){
					return input;
				}
				result += baseCompute(pre, "", "ln") + "*" + input;
				return result;
			}
			else {
				result = "Error!";
				return result;
			}
		}
		// ��lnx��
		if ((tmpIndex = (int)input.find("ln")) != -1){
			string back = input.substr(tmpIndex + 2, n - tmpIndex-1);
			if (back == flag){
				result = "1.0/" + flag;
				return result;
			}
			else if(back!=flag){
				result = "Error!";
				return result;
			}
		}
		
		// ��cosx��
		if ((tmpIndex = (int)input.find("cos")) != -1){
			cout << tmpIndex << endl;
			string back = input.substr(tmpIndex + 3, n - tmpIndex - 3);
			if (back == flag){
				result = "-1.0*sin" + flag;
				return result;
			}
			else if (back != flag){
				result = "Error!";
				return result;
			}
		}
	}

}

// ��������
double baseCompute(double A, double B, string flag){
	if (flag == "+"){
		return (A + B);
	}
	else if (flag == "-"){
		return (A - B);
	}
	else if (flag == "/"){
		return (A / B);
	}
	else if (flag == "*"){
		return (A*B);
	}
	else if (flag == "sin"){
		return (sin(A));
	}
	else if (flag == "cos"){
		return (cos(A));
	}
	else if (flag == "tan"){
		return (tan(A));
	}
	else if (flag == "log"){
		return (log(A) / log(B));
	}
	else if (flag == "ln"){
		return (log(A) / log(NATURAL_LOGARITHM));
	}
	else if (flag == "exp"){
		return (exp(A));
	}
	else if (flag == "^"){
		return (pow(A, B));
	}
	return 0;
}

// ��������
string baseCompute(string strA, string strB, string flag){
	double A = string2double(strA);
	double B = string2double(strB);	
	string result = double2string(baseCompute(A, B, flag), 15);
	return result;
}

// ������ʽ
bool simplifyEquation(string& content, string flag, string& minus_mask){
//	std::cout << "--> simplifyEquation  " << content << "  flag = " << flag << "   mask = "<<minus_mask<<endl;
	//cout << "minus_mask = " << minus_mask << endl;
	CV_Assert(flag == "*/" || flag == "^" || flag == "+-" || flag == "spe");
	int index_exp = -1;
	
	// ��һ�������������ı�ţ� �ڶ���int�����������ֵ�λ��
 	vector<pair<int,int>> index;
	
	string tmpFlag = "";

	string old_content = content;

	// ��ʱ����
	int tmp_index = 0;

	// ��ֹ���� ��������������������
	bool suspend = false;

	while (1){
		index_exp = -1;
		suspend = false;
		index.clear();
		if (flag == "^")
		{
			index_exp = (int)content.find_first_of(flag, index_exp + 1);
			tmpFlag = flag;
		}

		else if (flag == "*/"){
			tmp_index = (int)content.find_first_of("*");
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(MUL_INDEX, 1*tmp_index));
			tmp_index = (int)content.find_first_of("/");
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(DIV_INDEX, 1 * tmp_index));			

			if (index.size() > 0){
				index_exp = minPair(index, tmp_index);
				tmpFlag = ComputeType[tmp_index];
			}			
		}
		else if (flag == "+-"){
			tmp_index = (int)content.find_first_of("+");
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(ADD_INDEX, 1 * tmp_index));
			tmp_index = (int)content.find_first_of("-");
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(MNS_INDEX, 1 * tmp_index));

			
			if (index.size() > 0){
				index_exp = minPair(index, tmp_index);
				//cout << ComputeType[tmp_index] << endl;
				tmpFlag = ComputeType[tmp_index];
			}
		}
		else if (flag == "spe"){
			tmp_index = (int)content.find("sin");
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(SIN_INDEX, 1 * tmp_index));
			tmp_index = (int)content.find("cos");
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(COS_INDEX, 1 * tmp_index));
			tmp_index = (int)content.find("tan");
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(TAN_INDEX, 1 * tmp_index));

			if (index.size() > 0){
				index_exp = minPair(index, tmp_index) + 2;
				tmpFlag = ComputeType[tmp_index];
			}
		}

		if (tmpFlag == "-" && index_exp == 0){
			index.clear();
			tmp_index = (int)content.find_first_of("+", 1);
			//cout << "tmp_index =  " << tmp_index << endl;
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(ADD_INDEX, 1 * tmp_index));
			tmp_index = (int)content.find_first_of("-", 1);
			if (tmp_index != -1)	index.push_back(make_pair<int, int>(MNS_INDEX, 1 * tmp_index));

			//cout <<"tmp_index =  "<< tmp_index << endl;

			/*for (int j = 0; j<index.size(); j++){
				cout << "first = " << index[j].first << "  second = " << index[j].second << endl;
			}*/

			if (index.size() > 0){
				index_exp = minPair(index, tmp_index);
				tmpFlag = ComputeType[tmp_index];
				//cout  << tmpFlag << endl;
			}
		}

		if (index_exp == -1)
			break;

		string pre_str;
		string back_str;

		int pre = index_exp;
		int back = index_exp;

		// ����������
		for (int i = index_exp + 1; i < content.length(); i++){
			if (i == index_exp + 1 && content[i] == '-')
				back++;
			else if ((content[i] >= '0' && content[i] <= '9') || content[i] == '.')
				back++;
			else
				break;
		}
		// ǰ��������
		for (int i = index_exp - 1; i >= 0; i--){
			if ((content[i] >= '0' && content[i] <= '9') || content[i] == '.')
				pre--;
			else if (content[i] == '-'){
				if (tmpFlag == "^" || flag == "*/"){
					if (minus_mask[i] == '0'){
						break;
					}
					else if (minus_mask[i] == '1'){
						pre--;
						break;
					}
				}
				else if (tmpFlag == "-" && minus_mask[i] == '0' && i == index_exp-1){
					content.replace(index_exp - 1, 2, "+");
					suspend = true;
					break;
					index_exp = index_exp - 1;
				}
				else {
					pre--;
					break;
				}
			}
			else
				break;
		}

		if (suspend)
			continue;

		pre_str = content.substr(pre, index_exp - pre);
		back_str = content.substr(index_exp + 1, back - index_exp);

		/*cout << "content: " << content << endl;
		if (flag == ""){
			cout << pre_str << tmpFlag << back_str << " index =  " << index_exp <<endl;
		}*/

		if (flag != "spe"){
			string strT = baseCompute(pre_str, back_str, tmpFlag);
			string tmp_zero_str(strT.length(), '0');
			content.replace(pre, back - pre + 1, strT);
			minus_mask.replace(pre, back - pre + 1, tmp_zero_str);
			if (string2double(strT) < 0)
				minus_mask[pre] = '1';
		}
		else {
			string strT = baseCompute(back_str, "", tmpFlag);
			string tmp_zero_str(strT.length(), '0');
			content.replace(index_exp - 2, back - index_exp + 3, strT);
			minus_mask.replace(pre, back - pre + 1, tmp_zero_str);
		}
		
		//std::cout <<tmpFlag<<" " << "\tcontent " << content << "\t\t mask = "<<minus_mask<<endl;

		if (old_content == content)
			break;
		old_content = content;
	}
	//cout << "content = " << content << "        mask = "<<minus_mask<<endl;
	return true;
}

//// ������ʽ
//bool simplifyEquation(string& content, string flag){
//	std::cout << "--> simplifyEquation  " <<content<<"  flag = "<<flag<<endl;
//	CV_Assert(flag == "*/" || flag == "^" || flag == "+-" || flag == "spe");
//	int index_exp = -1;
//	int index_1 = -1;
//	int index_2 = -1;
//	int index_3 = -1;
//	string tmpFlag = "";
//
//	string old_content = content;
//
//	while (1){
//		index_exp = -1;
//		if (flag == "^")
//		{
//			index_exp = content.find_first_of(flag, index_exp + 1);
//			tmpFlag = flag;
//		}
//		
//		else if (flag == "*/"){
//			index_1 = content.find_first_of("*");
//			index_2 = content.find_first_of("/");
//
//			if (index_1 != -1 && index_2 == -1){
//				index_exp = index_1;
//				tmpFlag = "*";
//			}
//			else if (index_1 == -1 && index_2 != -1){
//				index_exp = index_2;
//				tmpFlag = "/";
//			}
//			else if (index_1 != -1 && index_2 != -1){
//				index_exp = min(index_1, index_2);
//				if (index_exp == index_1)
//					tmpFlag = "*";
//				else
//					tmpFlag = "/";
//			}
//		}
//		else if (flag == "+-"){
//			index_1 = content.find_first_of("+");
//			index_2 = content.find_first_of("-");
//			//printf("index_1 = %d index_2 = %d\n", index_1, index_2);
//			if (index_1 != -1 && index_2 == -1){
//				index_exp = index_1;
//				tmpFlag = "+";
//			}
//			else if (index_1 == -1 && index_2 != -1){
//				index_exp = index_2;
//				tmpFlag = "-";
//			}
//			else if (index_1 != -1 && index_2 != -1){
//				index_exp = min(index_1, index_2);
//				if (index_exp == index_1)
//					tmpFlag = "+";
//				else
//					tmpFlag = "-";
//			}
//		}
//
//		else if (flag == "spe"){
//			index_1 = content.find("sin");
//			index_2 = content.find("cos");
//			//index_3 = content.find_first_of("log");
//			
//			//printf("index_1 = %d index_2 = %d\n", index_1, index_2);
//			if (index_1 != -1 && index_2 == -1){
//				index_exp = index_1+2;
//				tmpFlag = "sin";
//			}
//			else if (index_1 == -1 && index_2 != -1){
//				index_exp = index_2+2;
//				tmpFlag = "cos";
//			}
//			else if (index_1 != -1 && index_2 != -1){
//				index_exp = min(index_1, index_2);
//				if (index_exp == index_1)
//					tmpFlag = "sin";
//				else
//					tmpFlag = "cos";
//				index_exp += 2;
//			}
//		}
//
//		//cout << "tmpFlag = " << tmpFlag << endl;
//
//		if (tmpFlag == "-" && index_exp == 0){
//			cout << "hello" << endl;
//			index_1 = content.find_first_of("+", 1);
//			index_2 = content.find_first_of("-", 1);
//			//printf("index_1 = %d index_2 = %d\n", index_1, index_2);
//			if (index_1 != -1 && index_2 == -1){
//				index_exp = index_1;
//				tmpFlag = "+";
//			}
//			else if (index_1 == -1 && index_2 != -1){
//				index_exp = index_2;
//				tmpFlag = "-";
//			}
//			else if (index_1 != -1 && index_2 != -1){
//				index_exp = min(index_1, index_2);
//				if (index_exp == index_1)
//					tmpFlag = "+";
//				else
//					tmpFlag = "-";
//			}
//		}
//
//		if (index_exp == -1)
//			break;
//
//		string pre_str;
//		string back_str;
//
//		int pre = index_exp;
//		int back = index_exp;
//
//		for (int i = index_exp + 1; i < content.length(); i++){
//			if (i == index_exp + 1 && content[i] == '-')
//				back++;
//			else if ((content[i] >= '0' && content[i] <= '9') || content[i] == '.')
//				back++;
//			else
//				break;
//		}
//		for (int i = index_exp - 1; i >= 0; i--){
//			if ((content[i] >= '0' && content[i] <= '9') || content[i] == '.')
//				pre--;
//			else if (content[i] == '-'){
//				if (tmpFlag == "^"){
//					if (i - 1 >= 0){
//						if ((content[i - 1] >= '0' && content[i - 1] <= '9') || content[i] == '.'){
//							break;
//						}
//					}
//				}
//				else {
//					pre--;
//					break;
//				}
//			}
//			else 
//				break;
//		}
//
//		pre_str = content.substr(pre, index_exp - pre);
//		back_str = content.substr(index_exp + 1, back - index_exp);
//
//		if (flag != "spe"){
//			string strT = baseCompute(pre_str, back_str, tmpFlag);
//			content.replace(pre, back - pre + 1, strT);
//		}
//		else {
//			string strT = baseCompute(back_str, "", tmpFlag);
//			content.replace(index_exp-2, back - index_exp + 3, strT);
//		}
//		std::cout << "\tcontent " << content << endl;
//
//		if (old_content == content)
//			break;
//		old_content = content;
//	}
//	
//	return true;
//}

// ������ʽ��ֵ
double calculateEquation(string equation){
	//
	//cout << "ԭʼ��ʽΪ:" << endl << "\t" << equation << endl;
	removeAllSpecChar(equation, ' ');
	string minus_mask(equation.length(), '0');
	while (1){
		bool bSuc = simplifyRightestBrackets(equation, minus_mask);
		//cout << endl << "���� : " << equation << endl;
		//cout << "���� : " << minus_mask << endl;
		if (!bSuc)
			break;
	}
	//return 0;
	return calcEquationWithoutBrackets(equation, minus_mask);
}

// �ж�string��ĳһλ���Ƿ���������
bool isInBracket(string input, int pos){
	int lbCount = 0, rbCount = 0;
	if (input[pos] == ')' || input[pos] == '(')
		return true;
	for (int i = 0; i < pos; i++){
		if (input[i] == ')')
			rbCount++;
		else if (input[i] == '(')
			lbCount++;
	}
	return (lbCount != rbCount);
}

// �滻string input������string AΪstring B
bool replaceAll(string& input, string SearchTerm, string ReplacementTerm){
	ReplacementTerm = "(" + ReplacementTerm + ")";
	string::size_type pos = 0;
	string::size_type srcLen = SearchTerm.size();
	string::size_type desLen = ReplacementTerm.size();
	pos = input.find(SearchTerm, pos);
	while ((pos != string::npos))
	{
		input.replace(pos, srcLen, ReplacementTerm);
		pos = input.find(SearchTerm, (pos + desLen));
	}
	return true;
}

// ��������
string substituteVariables(string& input, string SearchTerm, string ReplacementTerm){
	ReplacementTerm = "(" + ReplacementTerm + ")";
	string monomial(input);
	
	string::size_type pos = 0;
	string::size_type srcLen = SearchTerm.size();
	string::size_type desLen = ReplacementTerm.size();
	pos = monomial.find(SearchTerm, pos);
	while ((pos != string::npos))
	{
		//cout << pos << endl;
		string tmp = monomial.substr(pos, 3);
		if (isSpecialString(tmp)){
			pos = monomial.find(SearchTerm, pos + 3);
			continue;
		}

		monomial.replace(pos, srcLen, ReplacementTerm);
		pos = monomial.find(SearchTerm, (pos + desLen));
	}
	return monomial;
}


// �ж��Ƿ�Ϊ�����ַ���
bool isSpecialString(string input){
	bool flag = false;
	for (int i = 0; i < COMPUTE_TYPE_COUNT; i++){
		if (input == ComputeType[i])
			return true;
	}
	return flag;
}

// ���㵥��ʽ��ֵ
double calculateMonomial(string monomial, double value, string unknownNum){
	string equation = substituteVariables(monomial, unknownNum, double2string(value, 15));
	double result = calculateEquation(equation);
	cout << "phi(" << value << ") = " << result << endl;
	return result;
}

// �������ʽ��ֵ������ʽ������ x[0], x[1], x[2]����ʽ
double calculatePolynomialToSingleValue(string polynomial, Mat input){
	double* data = input.ptr<double>(0);
	string result(polynomial);
	for (int i = 0; i < input.rows*input.cols; i++){
		replaceAll(result, "x[" + int2string(i) + "]", "(" + double2string(data[i], 15) + ")");
		//cout << result << endl;
	}
	//cout << "result = " << result << endl;
	double r = calculateEquation(result);
	//cout <<"r = " << r << endl;
	return r;
}

// ���������ʽ����ʽ��ֵ ����ʽ Hess = [x1+x2, x1-x2;x1 ,x2]; �ɶ��źͷֺž���ʽ������
Mat calculatePolynomialInMat(vector<vector<string>> matPolynomial, Mat input){
	Mat result = Mat::zeros((int)matPolynomial.size(), (int)matPolynomial[0].size(), CV_64FC1);
	//cout << "input = " << input << endl;
	cout << "result.size = " << result.size() << endl;
	for (int i = 0; i < result.rows; i++){
		cout << "i = " << i << endl;
		double* data = result.ptr<double>(i);
		for (int j = 0; j < result.cols; j++){
			*data++ = calculatePolynomialToSingleValue(matPolynomial[i][j], input);
			//cout << "data[" << i * 100 + j << "] = " << *data << endl;
		}
	}

	return result;
}

// ���������ʽ����ʽ��ֵ ����ʽ Hess = [x1+x2, x1-x2;x1 ,x2]; �ɶ��źͷֺž���ʽ������
Mat calculatePolynomialInMat(string matPolynomial, Mat input){
	vector<vector<string>> polynomials = splitPolynomialInMat(matPolynomial);

	/*for (int i = 0; i < polynomials.size(); i++){
	cout << polynomials[i] << "   ";
	}
	cout << endl;*/

	Mat result = Mat::zeros((int)polynomials.size(), (int)polynomials[0].size(), CV_64FC1);

	for (int i = 0; i < result.rows; i++){
		double* data = result.ptr<double>(i);
		for (int j = 0; j < result.cols; j++){
			*data++ = calculatePolynomialToSingleValue(polynomials[i][j], input);
		}
	}

	return result;
}

// �������ʽ��ֵ������ʽ������ x[0], x[1], x[2]����ʽ
Mat calculatePolynomial(vector<string> polynomial, Mat input){
	Mat result = Mat::zeros((int)polynomial.size(), 1, CV_64FC1);
	double* data = result.ptr<double>(0);

	for (int i = 0; i < polynomial.size(); i++){
		data[i] = calculatePolynomialToSingleValue(polynomial[i], input);
	}

	return result;
}

// �������ʽ��ֵ������ʽ������ x[0], x[1], x[2]����ʽ
Mat calculatePolynomial(string equations, Mat input){	
	vector<string> polynomials = splitManyPolynomialToSingleValue(equations);

	/*for (int i = 0; i < polynomials.size(); i++){
		cout << polynomials[i] << "   ";
	}
	cout << endl;*/

	Mat result = Mat::zeros((int)polynomials.size(), 1, CV_64FC1);
	double* data = result.ptr<double>(0);
	
	for (int i = 0; i < polynomials.size(); i++){
		data[i] = calculatePolynomialToSingleValue(polynomials[i], input);
	}

	return result;
}

// ��������ʽ
vector<string> splitManyPolynomialToSingleValue(string input){
	vector<string> result;
	if (input[0] != '['){
		result.push_back(input);
	}
	else {
		string::size_type pos = 1;
		string::size_type old_pos = 1;
		pos = input.find(';', pos);

		while ((pos != string::npos))
		{
			//cout << "pos = "<<pos << endl;
			result.push_back(input.substr(old_pos, pos - old_pos));
			old_pos = pos + 1;
			pos = input.find(';', pos+1);
		}
		result.push_back(input.substr(old_pos, input.length() - 1 - old_pos));
	}
	return result;
}

// ��������ʽ
vector<vector<string>> splitPolynomialInMat(string input){
	vector<vector<string>> result;
	if (input[0] != '['){
		vector<string> tmp;
		tmp.push_back(input);
		result.push_back(tmp);
		return result;
	}
	else {
		string::size_type pos = 1;
		string::size_type old_pos = 1;
		vector<string> rowStrings;
		pos = input.find(';', pos);

		while ((pos != string::npos))
		{
			//cout << "pos = "<<pos << endl;
			rowStrings.push_back(input.substr(old_pos, pos - old_pos));
			old_pos = pos + 1;
			pos = input.find(';', pos + 1);
		}
		rowStrings.push_back(input.substr(old_pos, input.length() - 1 - old_pos));

		// �ȰѸ����ҵ����ٷֿ���ϸ��
		
		vector<string> items;
		for (int i = 0; i < rowStrings.size(); i++){
			items.clear();
			pos = 0;
			old_pos = 0;
			pos = rowStrings[i].find(',', pos);

			while ((pos != string::npos)){
				items.push_back(rowStrings[i].substr(old_pos, pos - old_pos));
				old_pos = pos + 1;
				pos = rowStrings[i].find(',', pos + 1);
			}
			items.push_back(rowStrings[i].substr(old_pos, rowStrings[i].length() - old_pos));
			result.push_back(items);
		}
	}
	return result;
}

// �ж��ַ���string�Ƿ�������
bool isnum(string s)
{
	int flag = (int)s.find(' ');					// ���ڿո���Ϊ�������֣�����
	if (flag != -1)
		return false;
	stringstream sin(s);
	double t;
	char p;
	if (!(sin >> t))
		/*���ͣ�
		sin>>t��ʾ��sinת����double�ı�������ʵ����int��float�͵Ķ�����գ������ת���ɹ�����ֵΪ��0�����ת�����ɹ��ͷ���Ϊ0
		*/
		return false;
	if (sin >> p)
		/*���ͣ��˲������ڼ����������У����ּ��ַ�����������ʽ�����磺34.f����������ĵĲ��֣�sin>>t���Ѿ����ղ�ת������������ֲ��֣���stringstream����ӦҲ�����һ���ָ�����������ʱ�����ַ��������ּ��ַ�����������ʽ����˲��ֿ���ʶ�𲢽����ַ����֣�����������˵�ģ����յ���.f�ⲿ�֣�������������������false;���ʣ�µĲ��ֲ����ַ�����ô��sin>>p��Ϊ0,����е���һ��else����
		*/
		return false;
	else
		return true;
}

// �ַ����ϲ�
string mergeVecStr(vector<string> strs){
	string result = "";
	vector<string>::iterator it = strs.begin();
	while (it!=strs.end())
	{
		result += *it;
		it++;
	}
	return result;
}

// char���ַ� ת string
string char2string(char c){
	string result;
	result.resize(1);
	result[0] = c;
	return result;
}

// ��׺���ʽת��Ϊ��׺���ʽ infixExpΪ�����ŵ���������
string convert2PostfixExp(string infixExp){
	vector<string> strs;
	stack<char> chs;
	//9 + (3 - 1) * 3 + 10 / 2
	for (int i = 0; i < (int)infixExp.size();){
		string tmp = findNum(infixExp, i);
		int len = (int)tmp.size();
		if (len == 0){					// ��ǰΪ����ʱ
			char c = infixExp[i];		// ��ȡ��ǰ����
			if (chs.empty()){			// ��Ϊ��ջ��ֱ��push
				chs.push(c);
			}
			else{
				if (c == ')'){
					while (!chs.empty() && chs.top() != '('){
						strs.push_back(char2string(chs.top()) + " ");
						chs.pop();
					}
					chs.pop();			// ��'('Ҳ��pop��
					i++;
					continue;
				}
				bool flag = (((c == '*' || c == '/') && (chs.top() != '*' && chs.top() != '/'))									// ��ǰԪ�ص����ȼ�����ջ��Ԫ��
					|| ((c == '+' || c == '-') && chs.top() == '('));	// ��ǰԪ�ص����ȼ�����ջ��Ԫ��

				if (c == '(' || flag){
					// ��ǰ�����Ϊ�����Ż������ȼ�����ջ��Ԫ��
					chs.push(infixExp[i]);
				}
				else if (!flag){										// ��ǰԪ�ص����ȼ�������ջ��Ԫ��
					while (chs.top() != '('){
						//cout << "hellow mmd" << endl;
						strs.push_back(char2string(chs.top()) + " ");
						chs.pop();
						if (chs.empty()) break;							// ��ջΪ�գ���ѭ���˳�
					}
					chs.push(c);
				}
			}
			i++;
		}
		else if(len > 0){				// ��ǰΪ����ʱ
			strs.push_back(infixExp.substr(i, len)+" ");
			i = i + len;
		}
	}
	while (!chs.empty()){
		strs.push_back(char2string(chs.top()) + " ");
		chs.pop();
	}
	string result = mergeVecStr(strs);
	result.erase(result.end()-1);
	cout << "result = " << result << endl;
	return result;
}

// ���ַ���str�У��ӵ�idxλ�ÿ�ʼ�����ң��ҵ���Ŀ��ܹ������ֵ��Ӵ�
string findNum(string str, int idx){
	CV_Assert(idx >= 0 && idx < (int)str.length());
	int max_Len = (int)str.length() - idx;

	for (int i = 0; i < max_Len; i++){
		if (i == max_Len - 1){					// �жϵ������һ���ַ�
			string result = str.substr(idx, max_Len);
			if (isnum(result))
				return result;
		}
		if (!isnum(str.substr(idx, i + 1))){
			return str.substr(idx, i);
		}
	}	
	return "";
}

// ��׺���ʽ����ֵ
double calculatePostfixExp(string postfixExp){
	stack<double> d;
	for (int i = 0; i < (int)postfixExp.size();){
		string tmp = findNum(postfixExp, i);
		int len = (int)tmp.size();
		if (len == 0){					// ��ǰΪ����ʱ
			char c = postfixExp[i];		// ��ȡ��ǰ����
			double b = d.top();
			d.pop();
			double a = d.top();
			d.pop();
			switch (c)
			{
			case '+':d.push(a + b); break;
			case '-':d.push(a - b); break;
			case '*':d.push(a * b); break;
			case '/':d.push(a / b); break;
			default:
				break;
			}
			i = i + 2;
		}
		else {
			d.push(string2double(tmp));
			i = i + len+1;
		}
	}
	return d.top();
}


// Add 20170617
// KMPģʽƥ���㷨ʵ�� �������ݽṹ��P141
vector<int> get_next(string T){
	int i = 0;
	int j = -1;
	vector<int> next(T.size());
	next[0] = -1;
	while (i < (int)T.size()){
		if (j == -1 || T[i] == T[j]){	// T[i] ��ʾ��׺�ĵ����ַ���T[j]��ʾǰ׺�ĵ����ַ�
			++i;
			++j;
			next[i] = j;
		}
		else{
			j = next[j];				// ���ַ�����ͬ����jֵ����
		}
	}
	i = 0;
	while (i < (int)T.size()){
		next[i++]++;
	}
	
	return next;
}

// ����ӡ����ӡ������Ϣ Bool
void OutputDebugStringBool(bool flag, string name){
	CString m = string2CString(name);
	if (flag)
		OutputDebugString(m + _T(" is true!\r\n"));
	else
		OutputDebugString(m + _T(" is false!\r\n"));
}

// ����ӡ����ӡ������Ϣ Int
void OutputDebugStringInt(int v, string name){
	CString m = string2CString(name);
	CString str;
	str.Format(_T(" = %d\r\n"), v);
	m += str;
	OutputDebugString(m);
}

// ����ӡ����ӡ������Ϣ Double
void OutputDebugStringDouble(double v, string name){
	CString m = string2CString(name);
	CString str;
	str.Format(_T(" = %lf\r\n"), v);
	m += str;
	OutputDebugString(m);
}



// �ַ������� ������
//string string1 = "mark kwain";
//reverse(string1.begin(), string1.end());
// ************************************* //
// char array1[] = "mark twain";
// int N1 = strlen(array1);
// reverse(&array[0], &array1[N1]); 
/* ��ע�⣬&array1[N1]ʵ���������������һ���ַ�֮���Ǹ��ڴ浥Ԫ�ĵ�ַ����STL�㷨��Լ������������������first��last���ݸ��㷨ʱ��ͨ����Ϊ����
	�����˴������е����(first)���յ�(last), ������ȥ����lastָ���Ԫ��.*/