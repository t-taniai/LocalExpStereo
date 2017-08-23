#pragma once

#include <opencv2/core/core.hpp>
#include <fstream>

namespace cvutils
{
	namespace io
	{
		using byte = uchar;
		static int is_little_endian()
		{
			if (sizeof(float) != 4)
			{
				printf("Bad float size.\n"); exit(1);
			}
			byte b[4] = { 255, 0, 0, 0 };
			return *((float *)b) < 1.0;
		}

		static cv::Mat read_pfm_file(const std::string& filename)
		{
			int w, h;
			char buf[256];
			FILE *f;
			fopen_s(&f, filename.c_str(), "rb");
			if (f == NULL)
			{
				//wprintf(L"PFM file absent: %s\n\n", filename.c_str());
				return cv::Mat();
			}

			int channel = 1;
			fscanf(f, "%s\n", buf);
			if (strcmp(buf, "Pf") == 0) channel = 1;
			else if (strcmp(buf, "PF") == 0) channel = 3;
			else {
				printf(buf);
				printf("Not a 1/3 channel PFM file.\n");
				return cv::Mat();
			}
			fscanf(f, "%d %d\n", &w, &h);
			double scale = 1.0;
			fscanf(f, "%lf\n", &scale);
			int little_endian = 0;
			if (scale < 0.0)
			{
				little_endian = 1;
				scale = -scale;
			}
			size_t datasize = w*h*channel;
			std::vector<byte> data(datasize * sizeof(float));

			cv::Mat image = cv::Mat(h, w, CV_MAKE_TYPE(CV_32F, channel));

			// Adjust the position of the file because fscanf() reads too much (due to "\n"?)
			fseek(f, -(long)datasize * sizeof(float), SEEK_END);
			size_t count = fread((void *)&data[0], sizeof(float), datasize, f);
			if (count != datasize)
			{
				printf("Expected size %d, read size %d.\n", datasize, count);
				printf("Could not read ground truth file.\n");
				return cv::Mat();
			}
			int native_little_endian = is_little_endian();
			for (int i = 0; i < datasize; i++)
			{
				byte *p = &data[i * 4];
				if (little_endian != native_little_endian)
				{
					byte temp;
					temp = p[0]; p[0] = p[3]; p[3] = temp;
					temp = p[1]; p[1] = p[2]; p[2] = temp;
				}
				int jj = (i / channel) % w;
				int ii = (i / channel) / w;
				int ch = i % channel;
				image.at<float>(h - 1 - ii, jj*channel + ch) = *((float *)p);
			}
			fclose(f);
			return image;
		}

		static void save_pfm_file(const std::string& filename, const cv::Mat& image)
		{
			int width = image.cols;
			int height = image.rows;

			FILE *stream;
			fopen_s(&stream, filename.c_str(), "wb");
			if (stream == NULL)
			{
				wprintf(L"PFM file absent: %s\n\n", filename.c_str());
				return;
			}
			// write the header: 3 lines: Pf, dimensions, scale factor (negative val == little endian)
			int channel = image.channels();
			if (channel == 1)
				fprintf(stream, "Pf\n%d %d\n%lf\n", width, height, -1.0 / 255.0);
			else if (channel == 3)
				fprintf(stream, "PF\n%d %d\n%lf\n", width, height, -1.0 / 255.0);
			else {
				printf("Channels %d must be 1 or 3\n", image.channels());
				return;
			}


			// pfm stores rows in inverse order!
			int linesize = width*channel;
			std::vector<float> rowBuff(linesize);
			for (int y = height - 1; y >= 0; y--)
			{
				auto ptr = image.ptr<float>(y);
				auto pBuf = &rowBuff[0];
				for (int x = 0; x < linesize; x++)
				{
					float val = (float)(*ptr);
					pBuf[x] = val;
					ptr++;
					/*if (val > 0 && val <= 255)
					rowBuf[x] = val;
					else
					{
					printf("invalid: val %f\n", flo(x,y));
					rowBuf[x] = 0.0f;
					}*/
				}
				if ((int)fwrite(&rowBuff[0], sizeof(float), width, stream) != width)
				{
					printf("[ERROR] problem with fwrite.");
				}
				fflush(stream);
			}

			fclose(stream);
			return;
		}


		static bool writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat)
		{
			cv::Mat out = out_mat;
			if (!out.isContinuous())
				out = out.clone();

			if (!ofs.is_open()) {
				return false;
			}
			if (out.empty()) {
				int s = 0;
				ofs.write((const char*)(&s), sizeof(__int32));
				return true;
			}
			__int32 rows = out.rows;
			__int32 cols = out.cols;
			__int32 type = out.type();

			ofs.write((const char*)(&rows), sizeof(__int32));
			ofs.write((const char*)(&cols), sizeof(__int32));
			ofs.write((const char*)(&type), sizeof(__int32));
			ofs.write((const char*)(out.data), out.elemSize() * out.total());

			return true;
		}


		static bool saveMatBinary(const std::string& filename, const cv::Mat& output) {
			std::ofstream ofs(filename, std::ios::binary);
			return writeMatBinary(ofs, output);
		}


		static bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat, bool readHeader = true)
		{
			if (!ifs.is_open()) {
				return false;
			}

			if (readHeader)
			{
				__int32 rows, cols, type;
				ifs.read((char*)(&rows), sizeof(__int32));
				if (rows == 0) {
					return true;
				}
				ifs.read((char*)(&cols), sizeof(__int32));
				ifs.read((char*)(&type), sizeof(__int32));

				in_mat.release();
				in_mat.create(rows, cols, type);
			}
			ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());

			return true;
		}


		static bool loadMatBinary(const std::string& filename, cv::Mat& output, bool readHeader = true) {
			std::ifstream ifs(filename, std::ios::binary);
			return readMatBinary(ifs, output, readHeader);
		}

	}

	inline bool contains(const std::string& str1, const std::string& str2)
	{
		std::string::size_type pos = str1.find(str2);
		if (pos == std::string::npos)
		{
			return false;
		}
		return true;
	}

	static cv::Mat channelDot(const cv::Mat& m1, const cv::Mat& m2)
	{
		cv::Mat m1m2 = m1.mul(m2);
		m1m2 = m1m2.reshape(1, m1.rows*m1.cols);
		cv::Mat m1m2dot;
		cv::reduce(m1m2, m1m2dot, 1, cv::REDUCE_SUM);
		return m1m2dot.reshape(1, m1.rows);
	}

	static cv::Mat channelSum(const cv::Mat& m1)
	{
		cv::Mat m = m1.reshape(1, m1.rows*m1.cols);
		cv::reduce(m, m, 1, cv::REDUCE_SUM);
		return m.reshape(1, m1.rows);
	}

	static cv::Mat& integralFilter(cv::Mat& out, const cv::Mat& in, int r)
	{
		cv::boxFilter(in, out, -1, cv::Size(r * 2 + 1, r * 2 + 1), cv::Point(-1, -1), false, cv::BORDER_CONSTANT);
		return out;
	}

	static void duplicateChannels(cv::Mat& out, const cv::Mat& m, int cn = 3)
	{
		std::vector<cv::Mat> pch;
		for (int i = 0; i < cn; i++)
			pch.push_back(m);
		cv::merge(pch, out);
	}

	static cv::Vec3d getRandomUnitVector__(double thetaRange = CV_PI)
	{
		double z0 =  cos(thetaRange);
		double z = cv::theRNG().uniform(z0, 1.0);
		double phi = cv::theRNG().uniform(0.0, CV_PI*2.0);
		double sinT = sqrt(1 - z*z);
		double cosP = cos(phi), sinP = sin(phi);
		return cv::Vec3d(sinT * cosP, sinT * sinP, z);
	}
	static cv::Vec3d getRandomUnitVector(double thetaRange = CV_PI)
	{
		double theta = cv::theRNG().uniform(0.0, thetaRange);
		double phi = cv::theRNG().uniform(0.0, CV_PI*2.0);
		double cosT = cos(theta), sinT = sin(theta);
		double cosP = cos(phi), sinP = sin(phi);
		return cv::Vec3d(sinT * cosP, sinT * sinP, cosT);
	}

	static cv::Rect getLargerRect(cv::Rect rect, int margin)
	{
		return cv::Rect(rect.x - margin, rect.y - margin, rect.width + margin * 2, rect.height + margin * 2);
	}

	inline cv::Mat computeFlowError(cv::Mat motion)
	{
		cv::Mat err = motion.mul(motion);
		cv::reduce(err.reshape(1, err.size().area()), err, 1, cv::REDUCE_SUM);
		cv::sqrt(err, err);
		return err.reshape(1, motion.rows);
	}
	inline cv::Mat resizeMotionSafe(cv::Mat motion, cv::Size size, int method = cv::INTER_LINEAR)
	{
		if (motion.size() == size) return motion;
		cv::Mat inter_res;
		cv::Mat inter_nne;
		cv::resize(motion, inter_res, size, 0, 0, method);
		cv::resize(motion, inter_nne, size, 0, 0, cv::INTER_NEAREST);
		cv::Mat invalid = computeFlowError(inter_res - inter_nne) > 1.0;
		inter_nne.copyTo(inter_res);
		inter_res = inter_res.mul(cv::Scalar((double)size.width / motion.cols, (double)size.height / motion.rows));
		return inter_res;
	}

}