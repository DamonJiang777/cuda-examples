#ifndef COMMON_HELPER_STRING_H_
#define COMMON_HELPER_STRING_H_

#include <string.h>
#include <string>
#include <strings.h>

#ifndef STRNCASECMP
#define STRNCASECMP strncasecmp
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
      int         string_start = StringRemoveDelimiter('-', argv[i]);
      const char *string_argv  = &argv[i][string_start];

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
      int         string_start = StringRemoveDelimiter('-', argv[i]);
      const char *string_argv  = &argv[i][string_start];

      int length = static_cast<int>(strlen(string_ref));
      if (!STRNCASECMP(string_argv, string_ref, length))
      {
        if (length + 1 <= static_cast<int>(strlen(string_argv)))
        {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value        = atoi(&string_argv[length + auto_inc]);
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

#endif
