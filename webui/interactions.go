package webui

import (
	"errors"
	"strings"

	"github.com/gofiber/fiber/v2"
	"github.com/mudler/LocalAGI/core/interactions"
	"github.com/mudler/LocalAGI/core/state"
	"github.com/mudler/cogito"
)

func interactionError(c *fiber.Ctx, status int, err error) error {
	return c.Status(status).JSON(fiber.Map{"error": err.Error()})
}

func (a *App) AnswerInteraction(pool *state.AgentPool) fiber.Handler {
	return func(c *fiber.Ctx) error {
		agent := pool.GetAgent(c.Params("name"))
		if agent == nil {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "Agent not found"})
		}

		payload := struct {
			QuestionID string   `json:"question_id"`
			Selected   []string `json:"selected"`
			Text       string   `json:"text"`
		}{}
		if err := c.BodyParser(&payload); err != nil {
			return interactionError(c, fiber.StatusBadRequest, err)
		}
		if strings.TrimSpace(payload.QuestionID) == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "question_id is required"})
		}

		err := agent.Interactions().Answer(payload.QuestionID, cogito.UserAnswer{Selected: payload.Selected, Text: payload.Text})
		switch {
		case errors.Is(err, cogito.ErrQuestionNotFound):
			return interactionError(c, fiber.StatusNotFound, err)
		case errors.Is(err, cogito.ErrInvalidAnswer):
			return interactionError(c, fiber.StatusBadRequest, err)
		case err != nil:
			return interactionError(c, fiber.StatusInternalServerError, err)
		default:
			return c.JSON(fiber.Map{"status": "answer_received"})
		}
	}
}

func (a *App) DecidePlan(pool *state.AgentPool) fiber.Handler {
	return func(c *fiber.Ctx) error {
		agent := pool.GetAgent(c.Params("name"))
		if agent == nil {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "Agent not found"})
		}

		payload := struct {
			PlanID   string    `json:"plan_id"`
			Approved *bool     `json:"approved"`
			Subtasks *[]string `json:"subtasks"`
			Feedback string    `json:"feedback"`
		}{}
		if err := c.BodyParser(&payload); err != nil {
			return interactionError(c, fiber.StatusBadRequest, err)
		}
		if strings.TrimSpace(payload.PlanID) == "" {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "plan_id is required"})
		}
		if payload.Approved == nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "approved is required"})
		}

		err := agent.Interactions().Decide(payload.PlanID, *payload.Approved, payload.Subtasks, payload.Feedback)
		switch {
		case errors.Is(err, interactions.ErrPlanNotFound):
			return interactionError(c, fiber.StatusNotFound, err)
		case errors.Is(err, interactions.ErrInvalidPlanDecision):
			return interactionError(c, fiber.StatusBadRequest, err)
		case err != nil:
			return interactionError(c, fiber.StatusInternalServerError, err)
		default:
			return c.JSON(fiber.Map{"status": "plan_decision_received"})
		}
	}
}

func (a *App) PendingInteractions(pool *state.AgentPool) fiber.Handler {
	return func(c *fiber.Ctx) error {
		agent := pool.GetAgent(c.Params("name"))
		if agent == nil {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "Agent not found"})
		}
		return c.JSON(agent.Interactions().Pending(c.Query("conversation_id")))
	}
}
