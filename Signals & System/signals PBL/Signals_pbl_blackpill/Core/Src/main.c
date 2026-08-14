/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "usb_device.h"
#include "gpio.h"
#include "usbd_cdc_if.h"


/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
#define IMG_SIZE 16384
uint8_t image_in[IMG_SIZE];
uint8_t image_out[IMG_SIZE];
volatile uint8_t data_ready = 0;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
void run_clahe(uint8_t* img_in, uint8_t* img_out, int width, int height);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USB_DEVICE_Init();
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
		    if (data_ready) {
        // 1. Turn ON onboard LED (PC13 is Active Low)
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_RESET);
        
        // 2. Execute CLAHE Image Enhancement
        run_clahe(image_in, image_out, 128, 128);
        
        // 3. Send the processed image back to the laptop
        CDC_Transmit_FS(image_out, IMG_SIZE);
        
        // 4. Turn OFF onboard LED
        HAL_GPIO_WritePin(GPIOC, GPIO_PIN_13, GPIO_PIN_SET);
        
        // 5. Reset flag for next frame
        data_ready = 0;

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}
}
/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 25;
  RCC_OscInitStruct.PLL.PLLN = 192;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_3) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */
/* USER CODE BEGIN 4 */
void run_clahe(uint8_t* img_in, uint8_t* img_out, int width, int height) {
    int rows = 8;
    int cols = 8;
    int tile_h = 16;
    int tile_w = 16;
    float clip_limit = 4.0f;
    
    // Static allocation in SRAM (to avoid stack overflow)
    static uint8_t tile_luts[8][8][256];
    
    // 1. Precompute the LUT for each tile
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // Compute local histogram
            // Compute local histogram
					static uint16_t hist[256];
					memset(hist, 0, sizeof(hist));
            for (int y = r * tile_h; y < (r + 1) * tile_h; y++) {
                for (int x = c * tile_w; x < (c + 1) * tile_w; x++) {
                    uint8_t pixel_val = img_in[y * width + x];
                    hist[pixel_val]++;
                }
            }
            
            // Enforce clip limit and redistribute excess
            int limit = (int)(clip_limit * 256.0f / 256.0f); // 256 pixels in a 16x16 tile
            if (limit < 1) limit = 1;
            
            int excess = 0;
            for (int i = 0; i < 256; i++) {
                if (hist[i] > limit) {
                    excess += (hist[i] - limit);
                    hist[i] = limit;
                }
            }
            
            while (excess > 0) {
                int step = excess / 256;
                if (step < 1) step = 1;
                
                for (int i = 0; i < 256; i++) {
                    if (excess > 0 && hist[i] < limit) {
                        int available = limit - hist[i];
                        int to_add = (step < available) ? step : available;
                        if (to_add > excess) to_add = excess;
                        hist[i] += to_add;
                        excess -= to_add;
                    }
                }
            }
            
            // Compute CDF
            // Compute CDF
						static uint16_t cdf[256];
            cdf[0] = hist[0];
            for (int i = 1; i < 256; i++) {
                cdf[i] = cdf[i-1] + hist[i];
            }
            
            // Min-Max scaling CDF
            int cdf_min_val = 0;
            for (int i = 0; i < 256; i++) {
                if (cdf[i] > 0) {
                    cdf_min_val = cdf[i];
                    break;
                }
            }
            
            int denominator = 256 - cdf_min_val;
            for (int i = 0; i < 256; i++) {
                if (denominator == 0) {
                    tile_luts[r][c][i] = 0;
                } else {
                    float val = ((float)(cdf[i] - cdf_min_val) / denominator) * 255.0f;
                    tile_luts[r][c][i] = (uint8_t)(val + 0.5f);
                }
            }
        }
    }
    
    // Tile center coordinates
    float tc_y[8];
    float tc_x[8];
    for (int i = 0; i < 8; i++) {
        tc_y[i] = i * 16.0f + 7.5f;
        tc_x[i] = i * 16.0f + 7.5f;
    }
    
    // 2. Perform Bilinear Interpolation for each pixel
    for (int y = 0; y < height; y++) {
        // Find row interpolation parameters
        int r0, r1;
        float b;
        if (y < tc_y[0]) {
            r0 = 0; r1 = 0; b = 0.0f;
        } else if (y >= tc_y[7]) {
            r0 = 7; r1 = 7; b = 0.0f;
        } else {
            int k = (int)((y - tc_y[0]) / 16.0f);
            r0 = k;
            r1 = k + 1;
            b = (y - tc_y[k]) / 16.0f;
        }
        
        for (int x = 0; x < width; x++) {
            // Find col interpolation parameters
            int c0, c1;
            float a;
            if (x < tc_x[0]) {
                c0 = 0; c1 = 0; a = 0.0f;
            } else if (x >= tc_x[7]) {
                c0 = 7; c1 = 7; a = 0.0f;
            } else {
                int k = (int)((x - tc_x[0]) / 16.0f);
                c0 = k;
                c1 = k + 1;
                a = (x - tc_x[k]) / 16.0f;
            }
            
            uint8_t pixel_val = img_in[y * width + x];
            
            // Map pixel using the 4 surrounding tile LUTs
            uint8_t s_tl = tile_luts[r0][c0][pixel_val];
            uint8_t s_tr = tile_luts[r0][c1][pixel_val];
            uint8_t s_bl = tile_luts[r1][c0][pixel_val];
            uint8_t s_br = tile_luts[r1][c1][pixel_val];
            
            // Bilinear Interpolation
            float s_top = (1.0f - a) * s_tl + a * s_tr;
            float s_bottom = (1.0f - a) * s_bl + a * s_br;
            float enhanced = (1.0f - b) * s_top + b * s_bottom;
            
            img_out[y * width + x] = (uint8_t)(enhanced + 0.5f);
        }
    }
}
/* USER CODE END 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
