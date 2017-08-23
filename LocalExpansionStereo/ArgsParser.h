#pragma once
#include <vector>
#include <map>
#include <string>

class ArgsParser
{
	std::vector<std::string> args;
	std::map<std::string, std::string> argMap;

	void parseArgments()
	{
		for (int i = 0; i < args.size(); i++)
		{
			if (args[i][0] == '-')
			{
				std::string name(&args[i][1]);
				if (i + 1 < args.size())
				{
					argMap[name] = args[i + 1];
					i++;
					//std::cout << name << ": " << argMap[name] << std::endl;
				}
			}
		}
	}
	template <typename T>
	T convertStringToValue(std::string str) const
	{
		return (T)std::stod(str);
	}
	template <> float convertStringToValue(std::string str) const{ return std::stof(str); }
	template <> int convertStringToValue(std::string str) const{ return std::stoi(str); }
	template <> std::string convertStringToValue(std::string str)const { return str; }
	template <> bool convertStringToValue(std::string str) const
	{
		if (str == "true") return true;
		if (str == "false") return false;
		return convertStringToValue<int>(str) != 0;
	}

public:
	ArgsParser(){}
	ArgsParser(int argn, const char **args)
	{
		for (int i = 0; i < argn; i++) {
			this->args.push_back(args[i]);
		}
		parseArgments();
	}
	ArgsParser(const std::vector<std::string>& args)
	{
		this->args = args;
		parseArgments();
	}

	template <typename T>
	bool TryGetArgment(std::string argName, T& value) const
	{
		auto it = argMap.find(argName);
		if (it == argMap.end())
			return false;

		value = convertStringToValue<T>(it->second);
		return true;
	}
};
