
#include <freertos/FreeRTOS.h>
#include "driver/gpio.h"
#include "driver/ledc.h"

#define servoPin GPIO_NUM_4
#define ledcChannel LEDC_CHANNEL_0
#define deg180 2048
#define deg0 430

void servoInit() {
  // LEDC timer configuration (C++ aggregate initialization)
  ledc_timer_config_t ledc_timer = {};
  ledc_timer.speed_mode = LEDC_LOW_SPEED_MODE;
  ledc_timer.timer_num = LEDC_TIMER_0;
  ledc_timer.duty_resolution = LEDC_TIMER_14_BIT;
  ledc_timer.freq_hz = 50;
  ledc_timer.clk_cfg = LEDC_AUTO_CLK;
  ESP_ERROR_CHECK(ledc_timer_config(&ledc_timer));

  // LEDC channel configuration
  ledc_channel_config_t ledc_channel = {};
  ledc_channel.speed_mode = LEDC_LOW_SPEED_MODE;
  ledc_channel.channel = ledcChannel;
  ledc_channel.timer_sel = LEDC_TIMER_0;
  ledc_channel.intr_type = LEDC_INTR_DISABLE;
  ledc_channel.gpio_num = servoPin;
  ledc_channel.duty = deg180; // Start off
  ledc_channel.hpoint = 0;
  ESP_ERROR_CHECK(ledc_channel_config(&ledc_channel));
}

// alternates between 0 and 180 - 2048 is 180 degrees (counterclockwise max)
void app_main() {
  servoInit();
  while(1) {
    ledc_set_duty(LEDC_LOW_SPEED_MODE, ledcChannel, deg180);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, ledcChannel);
    printf("ccwMax\n");
    vTaskDelay(pdMS_TO_TICKS(1000));
    ledc_set_duty(LEDC_LOW_SPEED_MODE, ledcChannel, deg0);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, ledcChannel);
    vTaskDelay(pdMS_TO_TICKS(1000));
    printf("cwMax\n");
  }
}