#ifndef COMMON_HELPER_STRING_H_
#define COMMON_HELPER_STRING_H_

#include <fstream>
#include <string.h>
#include <string>
#include <strings.h>

#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
#endif

#ifndef STRCPY
#define STRCPY(sFilePath, nLenght, sPath) strcpy(sFilePath, sPath)
#endif

#ifndef FOPEN
#define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#endif

inline int StringRemoveDelimiter(char delimiter, const char *string)
{
  int string_start = 0;
  while (string[string_start] == delimiter)
  {
    string_start++;
  }
  if (string_start >= static_cast<int>(strlen(string) - 1))
  {
    return 0;
  }

  return string_start;
}

inline bool CheckCmdLineFlag(const int argc, const char **argv, const char *string_ref)
{
  bool b_found = false;

  if (argc >= 1)
  {
    for (int i = 1; i < argc; i++)
    {
      int string_start = StringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];

      const char *equal_pos = strchr(string_argv, '=');

      int argv_length =
          static_cast<int>(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

      int length = strlen(string_ref);
      if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length))
      {
        b_found = true;
        continue; // TODO(damonJiang): may break be ok?
      }
    }
  }
  return b_found;
}

inline int GetCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref)
{
  bool b_found = false;

  int value = -1;
  if (argc >= 1)
  {
    for (int i = 1; i < argc; i++)
    {
      int string_start = StringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];

      int length = static_cast<int>(strlen(string_ref));
      if (!STRNCASECMP(string_argv, string_ref, length))
      {
        if (length + 1 <= static_cast<int>(strlen(string_argv)))
        {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = atoi(&string_argv[length + auto_inc]);
        }
        else
        {
          value = 0;
        }

        b_found = true;
        continue; // TODO(damonJiang): may break be ok?
      }
    }
  }
  if (b_found)
  {
    return value;
  }
  else
  {
    return 0;
  }
}

//! Find the path for a file assuming that
//! files are found in the searchPath
//!
//! @return the path if succeed, otherwish 0
//! @param file_name name of the file
//! @param executable_path optional absolute path of the executable
inline char *sdkFindFilePath(const char *file_name, const char *executable_path)
{
  // <executable_name> defines a varaable that is replacd with the name of the
  // executable

  // Typical relative search paths to locate needed companion files(e.g. smaple
  // input data, or JIT source files). The origin for the relative search may be
  // the .exe file, a .bat file lanuch an .exe, a browser .exe launching the
  // .exe or .bat, etc

  const char *serach_path[] = {"./",
                               "./data/",

                               "./example/<executable_name>/"};

  // extract the executable name
  std::string executable_name;

  if (executable_path != 0)
  {
    executable_name = std::string(executable_path);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// only support linux
#else
    // linux & osx path delimiter
    size_t delimiter_pos = executable_name.find_last_of('/');
    executable_name.erase(0, delimiter_pos + 1);
#endif

    // loop over all search paths and return the first hit
    for (unsigned int i = 0; i < sizeof(serach_path) / sizeof(char *); ++i)
    {
      std::string path(serach_path[i]);
      size_t exectuable_name_pos = path.find("<executable_name>");

      // if there is executable_name variable in the serach_path
      // replace it with the value
      if (exectuable_name_pos != std::string::npos)
      {
        if (executable_path != 0)
        {
          path.replace(exectuable_name_pos, strlen("<executable_name>"), executable_name);
        }
        else
        {
          // skip
          continue;
        }
      }

      // test if the file exists
      path.append(file_name);
      FILE *fp;
      FOPEN(fp, path.c_str(), "rb");

      if (fp != NULL)
      {
        fclose(fp);
        // file found
        // returning an allocated array here for backwards compatibility reasons
        char *file_path = reinterpret_cast<char *>(malloc(path.length() + 1));
        STRCPY(file_path, path.length() + 1, path.c_str());
        return file_path;
      }

      if (fp)
      {
        fclose(fp);
      }
    }
  }

  // file not found
  printf("\nerror: sdkFindFilePath: file <%s> not found!\n", file_name);
  return 0;
}
#endif
