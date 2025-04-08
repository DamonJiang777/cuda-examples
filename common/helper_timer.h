#ifndef COMMON_HELPER_TIMER_H_
#define COMMON_HELPER_TIMER_H_

// Definition of the StopWatch Interface, this is used if we don't want to use
// the CUT functions But rather in a self contained class interface.
class StopWatchInterface
{
public:
  StopWatchInterface() {}
  virtual ~StopWatchInterface() {}

public:
  //! Start time measurement
  virtual void start() = 0;

  //! Stop time measurement
  virtual void stop() = 0;

  //! Reset time counters to zero
  virtual void reset() = 0;

  //! Time in msec. after start. If the stop watch is still running(i.e. there)
  //! was no call to stop()) then the elaspsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  virtual float getTime() = 0;

  //! Mean time to date based on the number of times the stopwatch has been
  //! _stopped_ (ie finish sessions) and the current total time
  virtual float getAverageTime() = 0;
};

// Begin Stopwatch timer class definitions for all OS platforms
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// only support linux
#else

// declarations for Stopwatch on Linux and Mac OSX
#include <ctime>
#include <sys/time.h>

class StopWatchLinux : public StopWatchInterface
{
public:
  StopWatchLinux()
      : start_time()
      , diff_time(0.0)
      , total_time(0.0)
      , running(false)
      , clock_sessions(0)
  {
  }

  virtual ~StopWatchLinux() {};

  //! start time measurement
  inline void start();

  //! stop time measurement
  inline void stop();

  //! reset time counters to zero
  inline void reset();

  //! Time in msec. after start. If the stop watch is still running(i.e. there)
  //! was no call to stop()) then the elaspsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  inline float getTime();

  //! Mean time to date based on the number of times the stopwatch has been
  //! _stopped_ (ie finish sessions) and the current total time
  inline float getAverageTime();

private:
  //! helper functions

  //! get difference between start time and current time
  inline float getDiffTime();

private:
  //! start of measurement
  struct timeval start_time;

  //! time difference between the last start and stop
  float diff_time;

  //! Total time difference between stops and starts
  float total_time;

  //! flag if the stop watch is running
  bool running;

  //! number of times clock has been started
  // ！and stopped to allow averaging
  int clock_sessions;
};

inline void StopWatchLinux::start()
{
  gettimeofday(&start_time, 0);
  running = true;
}

inline void StopWatchLinux::stop()
{
  diff_time = getDiffTime();
  total_time += diff_time;
  running = false;
  clock_sessions++;
}

inline void StopWatchLinux::reset()
{
  diff_time      = 0;
  total_time     = 0;
  clock_sessions = 0;

  if (running)
  {
    gettimeofday(&start_time, 0);
  }
}

inline float StopWatchLinux::getTime()
{
  float retval = total_time;
  if (running)
  {
    retval += getDiffTime();
  }
  return retval;
}

inline float StopWatchLinux::getAverageTime()
{
  return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}

inline float StopWatchLinux::getDiffTime()
{
  struct timeval t_time;
  gettimeofday(&t_time, 0);

  return static_cast<float>(1000.0 * (t_time.tv_sec - start_time.tv_sec)
                            + (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}
#endif

inline bool sdkCreateTimer(StopWatchInterface **timer_interface)
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  // only support linux os
#else
  *timer_interface = reinterpret_cast<StopWatchLinux *>(new StopWatchLinux());
#endif
  return (*timer_interface != NULL) ? true : false;
}

inline bool sdkDeleteTimer(StopWatchInterface **timer_interface)
{
  if (*timer_interface)
  {
    delete *timer_interface;
    *timer_interface = NULL;
  }

  return true;
}

inline bool sdkStartTimer(StopWatchInterface **timer_interface)
{
  if (*timer_interface)
  {
    (*timer_interface)->start();
  }

  return true;
}

inline bool sdkStopTimer(StopWatchInterface **timer_interface)
{
  if (*timer_interface)
  {
    (*timer_interface)->stop();
  }

  return true;
}

inline bool sdkResetTimer(StopWatchInterface **timer_interface)
{
  if (*timer_interface)
  {
    (*timer_interface)->reset();
  }

  return true;
}

inline float sdkGetAverageTimerValue(StopWatchInterface **timer_interface)
{
  if (*timer_interface)
  {
    return (*timer_interface)->getAverageTime();
  }
  else
  {
    return 0.0f;
  }
}

inline float sdkGetTimerValue(StopWatchInterface **timer_interface)
{
  if (*timer_interface)
  {
    return (*timer_interface)->getTime();
  }
  else
  {
    return 0.0f;
  }
}

#endif
